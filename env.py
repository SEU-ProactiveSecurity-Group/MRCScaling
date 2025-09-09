import json
import networkx as nx
from attacker import MultiRouteAttacker, RandomAttacker, YoYoAttacker
from defender import PdDefender
import math
import random
import numpy as np


class Env:
    def __init__(
        self,
        multi_action_dim=False,
        attacker_type="static",
        max_steps=50,
        log_console=False,
    ):
        self.graphpath = "social-network-workmodel.json"
        self.metric_factor = {
            "cpu": 0.001,  # 标准值为0.001，调小可以减少cpu_stress系数对最后cpu占用影响，即加大traffic对cpu占用影响
            "cpu_b": 10,
            "memory": 0.0000001,  # 标准值为0.0000001
            "memory_b": 5,
            "delay_process_cpu": 0.2,
            "delay_process_memory": 0.25,
            "transmission": 1,  # max_traffic设的比较大，暂时没有考虑max_traffic对traffic的fallback
            "queue_rate": 1.4,  # 每个服务的排队系数，导致能容忍流量（或cpu或memory）可以超出100%一点
        }
        self.num_metrics = 5
        self.metrics = ["cpu", "memory", "transmission_rate", "delay", "replicas"]
        self.num_hosts = 4
        # self.num_hosts = 50
        self.hosts = self.load_hosts(self.num_hosts)
        self.graph, self.routes = self.load_graph(self.graphpath)
        self.num_services = self.graph.number_of_nodes()
        self.services = list(self.graph.nodes)
        self.initial_deploy()
        self.cur_step = 0
        self.evaluation = {
            "dangerous_nodes": 0,
            "route_delay": {},
            "route_fail_rate": {},
            "num_replicas": 0,
            "unused_replicas": {},
            "consecutive_successes": 0,
        }
        self.for_evaluation = {}
        self.execute_penalties = {
            "add": 0,
            "delete": 0,
        }
        self.scale_nums = {"add": 0, "delete": 0}
        self.state_dims = (self.num_services, self.num_metrics)
        self.multi_action_dim = multi_action_dim
        if self.multi_action_dim:
            self.action_dims = [10, 10]
        else:
            # self.action_dims = [19]
            self.action_dims = [82]
        self.attacker_type = attacker_type
        if self.attacker_type == "static":
            self.attacker = MultiRouteAttacker(self)
        elif self.attacker_type == "random":
            self.attacker = RandomAttacker(self)
        elif self.attacker_type == "yoyo":
            self.attacker = YoYoAttacker(self)
        self.state = np.zeros(self.state_dims, dtype=int)
        self.defender = PdDefender(self)
        self.max_steps = max_steps
        self.log_console = log_console

        self.attacked_routes = {}

    def load_hosts(self, num_hosts):
        hosts = []
        for i in range(num_hosts):
            # replicas: [{"name": "s1", "num": 1}]
            host = {"name": f"h{i}", "cpu": 4000, "memory": 8192, "replicas": {}}
            hosts.append(host)
        return hosts

    def load_graph(self, graphpath):
        with open(graphpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.DiGraph()
        # add nodes
        idx = 0
        for service, detail in data.items():
            if service == "s0":
                continue
            internal_service = detail["internal_service"]

            # cpu_limits = "500m", remove "m" and convert to int
            def parse_cpu(s):
                if s.endswith("m"):
                    return int(s[:-1])
                return int(s)

            # memory_limits = "512Mi", remove "Mi" and convert to int, or "1Gi" to 1024
            def parse_memory(s):
                if s.endswith("Mi"):
                    return int(s[:-2])
                elif s.endswith("Gi"):
                    return int(s[:-2]) * 1024
                return int(s)

            graph.add_node(
                service,
                idx=idx,
                workers=detail["workers"],
                threads=detail["threads"],
                loader=internal_service["loader"],
                replicas=detail["replicas"],
                limits={
                    "cpu": parse_cpu(detail["cpu-limits"]),
                    "memory": parse_memory(detail["memory-limits"]),
                },
                requests={
                    "cpu": parse_cpu(detail["cpu-requests"]),
                    "memory": parse_memory(detail["memory-requests"]),
                },
                traffic=0,
            )
            idx += 1
        # add edges
        for service, detail in data.items():
            if service == "s0":
                continue
            for external_service in detail["external_services"]:
                for s in external_service["services"]:
                    graph.add_edge(
                        service, s, prob=external_service["probabilities"][s]
                    )
        # print(graph.nodes(data=True))
        # s0的外部服务就是每条路由的起点，获取路由和节点的映射关系
        routes = {}
        for s in data["s0"]["external_services"]:
            for ss in s["services"]:
                routes[ss] = []
        for s, route in routes.items():
            rs = [s]
            while rs and len(rs) > 0:
                r = rs.pop(0)
                if r in route:
                    continue
                route.append(r)
                for neighbor in graph.neighbors(r):
                    if neighbor not in route:
                        rs.append(neighbor)

        # print(routes)
        return graph, routes

    def initial_deploy(self):
        # deploy replica to host
        for service, detail in self.graph.nodes(data=True):
            for _ in range(detail["replicas"]):
                host = self.deploy_replica_to_host(service)
                if host is None:
                    # raise ValueError(f"No available host for service {service}")
                    print(f"No available host for service {service}")

    def deploy_replica_to_host(self, service):
        selected_host = None
        selected_consumed_cpu = None
        selected_consumed_memory = None
        for host in self.hosts:
            consumed_cpu = sum(
                [self.graph.nodes[s]["requests"]["cpu"] for s in host["replicas"]]
            )
            consumed_memory = sum(
                [self.graph.nodes[s]["requests"]["memory"] for s in host["replicas"]]
            )
            if (
                host["cpu"]
                >= consumed_cpu + self.graph.nodes[service]["requests"]["cpu"]
                and host["memory"]
                >= consumed_memory + self.graph.nodes[service]["requests"]["memory"]
            ):
                # select the host with the least resources used
                if selected_host is None or (
                    (selected_consumed_cpu * 2 + selected_consumed_memory)
                    > (consumed_cpu * 2 + consumed_memory)
                ):
                    selected_host = host
                    selected_consumed_cpu = consumed_cpu
                    selected_consumed_memory = consumed_memory
        if selected_host is None:
            # print(f"No available host for service {service}")
            print(self.hosts)
            return None
        # deploy the replica to the selected host
        if service not in selected_host["replicas"]:
            selected_host["replicas"][service] = 1
        else:
            selected_host["replicas"][service] += 1
        return selected_host["name"]

    def remove_replica_from_host(self, service):
        # 首先检测服务副本数是否大于1，如果是，则从宿主机中删除一个副本
        if self.graph.nodes[service]["replicas"] <= 1:
            # print(f"Cannot remove last replica of service {service}")
            return None
        selected_host = None
        selected_consumed_cpu = None
        selected_consumed_memory = None
        for host in self.hosts:
            # select the host with the most resources used
            consumed_cpu = sum(
                [self.graph.nodes[s]["requests"]["cpu"] for s in host["replicas"]]
            )
            consumed_memory = sum(
                [self.graph.nodes[s]["requests"]["memory"] for s in host["replicas"]]
            )
            if service in host["replicas"] and (
                selected_host is None
                or (
                    (selected_consumed_cpu * 2 + selected_consumed_memory)
                    < (consumed_cpu * 2 + consumed_memory)
                )
            ):
                selected_host = host
                selected_consumed_cpu = consumed_cpu
                selected_consumed_memory = consumed_memory
        if selected_host is None or service not in selected_host["replicas"]:
            # print(f"No available host for service {service}")
            print(self.hosts)
            return None
        # remove the replica from the selected host
        selected_host["replicas"][service] -= 1
        if selected_host["replicas"][service] == 0:
            del selected_host["replicas"][service]
        return selected_host["name"]

    def add_replica_to_host(self, service):
        host = self.deploy_replica_to_host(service)
        if host is None:
            # print(f"Add service {service} replica failed")
            self.execute_penalties["add"] += 1
            return None
        self.scale_nums["add"] += 1
        self.graph.nodes[service]["replicas"] += 1
        return host

    def delete_replica_from_host(self, service):
        host = self.remove_replica_from_host(service)
        if host is None:
            # print(f"Delete service {service} replica failed")
            self.execute_penalties["delete"] += 1
            return None
        self.scale_nums["delete"] += 1
        self.graph.nodes[service]["replicas"] -= 1
        return host

    def scale_replicas(self, scale1, scale2):
        self.execute_penalties = {
            "add": 0,
            "delete": 0,
        }
        self.scale_nums = {"add": 0, "delete": 0}
        self.cal_state_from_graph()
        for service_idx, metrics in enumerate(self.state):
            service = self.services[service_idx]
            # scale1 >= scale2
            if scale1 >= scale2:
                # scale down
                if metrics[0] < scale2 - 10 and metrics[1] < scale2 - 10:
                    scale_num = math.ceil(
                        self.graph.nodes[service]["replicas"]
                        * (1 - max(metrics[0], metrics[1]) / scale1)
                    )
                    for _ in range(scale_num):
                        host = self.delete_replica_from_host(service)
                # scale up
                elif metrics[0] > scale1 or metrics[1] > scale1:
                    scale_num = math.ceil(
                        self.graph.nodes[service]["replicas"]
                        * (max(metrics[0], metrics[1]) / scale2 - 1)
                    )
                    for _ in range(scale_num):
                        host = self.add_replica_to_host(service)
            # scale1 < scale2
            else:
                # scale up
                if metrics[0] > scale2 or metrics[1] > scale2:
                    scale_num = math.ceil(
                        self.graph.nodes[service]["replicas"]
                        * (max(metrics[0], metrics[1]) / scale1 - 1)
                    )
                    for _ in range(scale_num):
                        host = self.add_replica_to_host(service)

    def cal_cpu(self, loader, traffic, factor):
        cpu = loader["cpu_stress"]
        return (
            math.ceil(
                (cpu["range_complexity"][0] + cpu["range_complexity"][1])
                / 2
                * cpu["trials"]
                * factor["cpu"]
            )
            * traffic
            + factor["cpu_b"]
        )

    def cal_memory(self, loader, traffic, factor):
        memory = loader["memory_stress"]
        return (
            math.ceil(memory["memory_size"] * memory["memory_io"] * factor["memory"])
            * traffic
            + factor["memory_b"]
        )

    def cal_transmission_rate(self, loader, traffic, workers, threads, factor):
        return traffic

    def cal_max_traffic(self, loader, workers, threads, factor):
        return workers * threads * factor["transmission"] * factor["queue_rate"]

    def cal_delay(self, loader, traffic, workers, threads, factor, replicas):
        # 计算总请求处理时延，ms
        process_delay = math.ceil(
            1000.0
            * (
                factor["delay_process_cpu"] * self.cal_cpu(loader, traffic, factor)
                + factor["delay_process_memory"]
                * self.cal_memory(loader, traffic, factor)
            )
            / (workers * threads * replicas)
        )
        # # 计算总请求排队时延
        # queue_delay = factor["delay_queue"] * traffic["queue"]
        # # 每个包平均时延
        # delay = (
        #     process_delay * traffic["process"]
        #     + queue_delay * (traffic["process"] + traffic["queue"])
        # ) / (traffic["process"] + traffic["queue"] + 1e-6)
        # return delay
        return process_delay

    def traffic_fallback(self):
        # 计算被攻击路由中瓶颈指标和瓶颈节点
        max_route_metrics = {}
        for route in self.attacked_routes:
            max_route_metrics[route] = {
                "traffic": {"value": 0, "service": None, "fail_percent": 0.0},
                "cpu": {"value": 0, "service": None, "fail_percent": 0.0},
                "memory": {"value": 0, "service": None, "fail_percent": 0.0},
            }

        # todo: 这里计算traffic瓶颈有点问题，不等于traffic，用cal_transmission_rate也不对，因为出现了max_traffic<traffic的情况
        # 计算每个服务的最大流量，存每条路由对应最大流量和资源指标
        for service, detail in self.graph.nodes(data=True):
            # 最大流量
            routes = []
            for route in self.attacked_routes:
                if service in self.routes[route]:
                    routes.append(route)
            # print("Service in Routes", service, routes)
            if not routes or len(routes) <= 0:
                continue
            # cpu和memory占用
            cpu = self.cal_cpu(detail["loader"], detail["traffic"], self.metric_factor)
            memory = self.cal_memory(
                detail["loader"], detail["traffic"], self.metric_factor
            )
            if (
                cpu
                > self.graph.nodes[service]["limits"]["cpu"]
                * self.graph.nodes[service]["replicas"]
                * self.metric_factor["queue_rate"]
            ):
                fail_percent = (
                    cpu
                    - self.graph.nodes[service]["limits"]["cpu"]
                    * self.graph.nodes[service]["replicas"]
                    * self.metric_factor["queue_rate"]
                ) / cpu
                for route in routes:
                    if fail_percent > max_route_metrics[route]["cpu"]["fail_percent"]:
                        max_route_metrics[route]["cpu"]["service"] = service
                        max_route_metrics[route]["cpu"]["value"] = cpu
                        max_route_metrics[route]["cpu"]["fail_percent"] = fail_percent
            if (
                memory
                > self.graph.nodes[service]["limits"]["memory"]
                * self.graph.nodes[service]["replicas"]
                * self.metric_factor["queue_rate"]
            ):
                fail_percent = (
                    memory
                    - self.graph.nodes[service]["limits"]["memory"]
                    * self.graph.nodes[service]["replicas"]
                    * self.metric_factor["queue_rate"]
                ) / memory
                for route in routes:
                    if (
                        fail_percent
                        > max_route_metrics[route]["memory"]["fail_percent"]
                    ):
                        max_route_metrics[route]["memory"]["service"] = service
                        max_route_metrics[route]["memory"]["value"] = memory
                        max_route_metrics[route]["memory"][
                            "fail_percent"
                        ] = fail_percent

        # print("Max Route Metrics", max_route_metrics)
        # self.print_traffic()
        
        self.clear_and_load_normal_traffic()
        # print(route_traffic)

        for route, metrics in max_route_metrics.items():
            fail_percent = 0.0
            for m in metrics:
                if fail_percent < metrics[m]["fail_percent"]:
                    fail_percent = metrics[m]["fail_percent"]
            # print(
            #     f"route {route} reached bottleneck for {max_m} overhead {fail_percent}"
            # )
            self.evaluation["route_fail_rate"][route] = fail_percent
            self.load_route_traffic(
                route, self.attacked_routes[route] * (1 - fail_percent)
            )
        # self.print_traffic()

        # print(
        #     "max traffic",
        #     self.graph.nodes[route]["limits"]["traffic"]
        #     * self.graph.nodes[route]["replicas"],
        # )

    def cal_state_from_graph(self):
        for service, detail in self.graph.nodes(data=True):
            s_idx = detail["idx"]
            for m in self.metrics:
                m_idx = self.metrics.index(m)
                if m_idx < 0:
                    continue
                if m == "cpu":
                    cpu = self.cal_cpu(
                        detail["loader"], detail["traffic"], self.metric_factor
                    )
                    self.state[s_idx, m_idx] = min(
                        math.ceil(
                            cpu * 100 / (detail["limits"]["cpu"] * detail["replicas"])
                        ),
                        100,
                    )
                elif m == "memory":
                    memory = self.cal_memory(
                        detail["loader"], detail["traffic"], self.metric_factor
                    )
                    self.state[s_idx, m_idx] = min(
                        math.ceil(
                            memory
                            * 100
                            / (detail["limits"]["memory"] * detail["replicas"])
                        ),
                        100,
                    )
                elif m == "transmission_rate":
                    transmission_rate = self.cal_transmission_rate(
                        detail["loader"],
                        detail["traffic"],
                        detail["workers"],
                        detail["threads"],
                        self.metric_factor,
                    )
                    self.state[s_idx, m_idx] = transmission_rate
                elif m == "delay":
                    delay = self.cal_delay(
                        detail["loader"],
                        detail["traffic"],
                        detail["workers"],
                        detail["threads"],
                        self.metric_factor,
                        detail["replicas"],
                    )
                    self.state[s_idx, m_idx] = delay
                elif m == "replicas":
                    self.state[s_idx, m_idx] = detail["replicas"]

        # print("State:\n", self.state)

    def cal_evaluation_metrics(self):
        self.evaluation = {
            "dangerous_nodes": 0,
            "route_delay": {},
            "route_fail_rate": self.evaluation.get("route_fail_rate", {}),
            "num_replicas": 0,
            "unused_replicas": {},
            "consecutive_successes": self.evaluation.get("consecutive_successes", 0),
        }
        # 危险节点数
        dangerous_threshold = 80
        for i in range(self.num_services):
            if self.state[i, 0] > dangerous_threshold:
                self.evaluation["dangerous_nodes"] += 1
            elif self.state[i, 1] > dangerous_threshold:
                self.evaluation["dangerous_nodes"] += 1

        # 链路时延
        for route, services in self.routes.items():
            if self.graph.nodes[route]["traffic"] > 0:
                route_delay = 0.0
                for service in services:
                    detail = self.graph.nodes[service]
                    delay = self.cal_delay(
                        detail["loader"],
                        detail["traffic"],
                        detail["workers"],
                        detail["threads"],
                        self.metric_factor,
                        detail["replicas"],
                    )
                    route_delay += delay
                self.evaluation["route_delay"][route] = route_delay

        # 副本数
        for service, detail in self.graph.nodes(data=True):
            service_idx = detail["idx"]
            unused_replicas = (
                detail["replicas"]
                * (
                    1
                    - max(self.state[service_idx, 0], self.state[service_idx, 1]) / 100
                )
                - 1
            )
            self.evaluation["unused_replicas"][service] = max(unused_replicas, 0)
            self.evaluation["num_replicas"] += detail["replicas"]
        # print(self.evaluation)

    def print_traffic(self):
        for service, detail in self.graph.nodes(data=True):
            print(
                f"Service {service} index {detail['idx']} traffic: {detail['traffic']}, limit: {detail['limits'].get('traffic', 0)}, replicas: {detail['replicas']}"
            )

    def print_state(self):
        for i in range(self.num_services):
            print(
                f"{self.services[i]} : {self.state[i, 0]}, {self.state[i, 1]}, {self.state[i, 2]}, {self.state[i, 3]}"
            )

    def load_route_traffic(self, route, traffic):
        if route not in self.graph.nodes:
            print(f"Route {route} does not exist in the graph.")
            return

        attacked_nodes = set()
        rs = [[route, traffic, 1.0]]
        while rs and len(rs) > 0:
            r, t, p = rs.pop(0)
            if r in attacked_nodes:
                continue
            attacked_nodes.add(r)
            current_traffic = math.ceil(t * p)
            self.graph.nodes[r]["traffic"] += current_traffic
            # 按概率分配流量
            for neighbor in self.graph.neighbors(r):
                if neighbor not in attacked_nodes:
                    prob = self.graph[r][neighbor]["prob"]
                    if prob > 0:
                        rs.append([neighbor, current_traffic, prob])

    def clear_and_load_normal_traffic(self):
        # 载入正常流量，假设每个服务的流量都是5左右
        for service in self.graph.nodes:
            self.graph.nodes[service]["traffic"] = random.randint(20, 30)

    def consume_attack_traffic(self, consume_percent=0.5):
        self.attacked_routes = {
            route: math.ceil(traffic * consume_percent)
            for route, traffic in self.attacked_routes.items()
        }

    def reset(self):
        self.evaluation = {
            "dangerous_nodes": 0,
            "route_delay": {},
            "route_fail_rate": {},
            "num_replicas": 0,
            "unused_replicas": {},
            "consecutive_successes": 0,
        }
        self.cur_step = 0
        self.hosts = self.load_hosts(self.num_hosts)
        self.graph, self.routes = self.load_graph(self.graphpath)
        self.initial_deploy()

        self.attacker.reset()
        self.defender.reset()

        self.clear_and_load_normal_traffic()
        self.cal_state_from_graph()

        if self.log_console:
            print("\n====================New Episode====================")
            print("\n>>>>Reset>>>>")
            print("Traffic:\n")
            self.print_traffic()
            print("State:\n")
            self.print_state()

        return self.state.copy()

    def step(self, action):
        if self.log_console:
            print(f"\n>>>>Step{self.cur_step}>>>>")
            print("\n>>>>Action:\n", action)

        # 防御
        scale_info = self.defender.defend(action)

        if self.log_console:
            print("\n>>>>After defend>>>>")
            print("Hosts:\n", self.hosts)
            print("Traffic:\n")
            self.print_traffic()
            self.cal_state_from_graph()
            print("State:\n")
            self.print_state()

        # 因为不考虑队列，所以上轮流量消耗完毕
        self.clear_and_load_normal_traffic()
        # self.consume_attack_traffic(0.5)
        self.consume_attack_traffic(0)

        # 攻击，更新 state 为 next_state
        self.attacker.attack()

        if self.log_console:
            print("\n>>>>After attack>>>>")
            print("Attacked routes:\n", self.attacked_routes)
            print("Traffic:\n")
            self.print_traffic()
            self.cal_state_from_graph()
            print("State:\n")
            self.print_state()

        # 结算本轮攻防后流量状态
        self.traffic_fallback()

        if self.log_console:
            print("\n>>>>After traffic fallback>>>>")
            print("Attacked routes:\n", self.attacked_routes)
            print("Traffic:\n")
            self.print_traffic()

        # 更新 state
        self.cal_state_from_graph()

        if self.log_console:
            print("\n>>>>After cal state from graph>>>>")
            print("State:\n")
            self.print_state()

        # 计算评估指标
        self.cal_evaluation_metrics()

        if self.log_console:
            print("\n>>>>After cal evaluation metrics>>>>")
            print("Evaluation metrics:\n", self.evaluation)

        # 计算 reward
        # 安全惩罚
        danger_penalty = self.evaluation["dangerous_nodes"] * -5
        # danger_penalty = self.evaluation["dangerous_nodes"] * -1
        fail_rate_penalty = sum(self.evaluation["route_fail_rate"].values()) * -2
        # 相比于前一步的安全奖励
        # more_safe_reward = 0
        # if self.for_evaluation.get("dangerous_nodes", 0) > 0:
        #     more_safe_reward = (
        #         self.evaluation["dangerous_nodes"]
        #         - self.for_evaluation["dangerous_nodes"]
        #     ) * -10
        # 用户体验惩罚
        # delay_penalty = (
        #     sum(
        #         [
        #             delay
        #             for delay in self.evaluation["route_delay"].values()
        #             if delay > 2000
        #         ]
        #     )
        #     / 1000
        #     * -0.2
        # )
        # 资源损耗惩罚
        # replica_cost = -17 * sum(self.evaluation["unused_replicas"].values()) / self.evaluation["num_replicas"]
        replica_cost = -0.4 * sum(self.evaluation["unused_replicas"].values())
        # replica_cost = -0.04 * sum(self.evaluation["unused_replicas"].values())
        # 资源利用奖励
        # replica_reward = (
        #     self.evaluation["num_replicas"]
        #     - sum(self.evaluation["unused_replicas"].values())
        # ) * 0.1
        # 时间消耗惩罚
        time_cost = -3
        # 动作执行失败惩罚
        # 如果攻击流量大于0时候，进行缩容，则惩罚，若攻击流量等于0，进行缩容或无动作，则奖励
        # execute_penalty = -0.1 * (
        #     self.execute_penalties["add"] + self.execute_penalties["delete"]
        # )
        # 动作消耗惩罚
        # scale_cost = -0 * (self.scale_nums["add"] + self.scale_nums["delete"])
        # 安全奖励
        defense_success = (
            self.evaluation["dangerous_nodes"] == 0
            and sum(self.evaluation["route_fail_rate"].values()) < 1
        )
        if defense_success:
            safety_reward = 5
        else:
            safety_reward = 0

        self.for_evaluation = self.evaluation.copy()
        self.cur_step += 1

        # 连续成功步数
        if self.evaluation["consecutive_successes"] > 0 and defense_success:
            self.evaluation["consecutive_successes"] += 1
        elif self.evaluation["consecutive_successes"] < 0 and not defense_success:
            self.evaluation["consecutive_successes"] -= 1
        else:
            self.evaluation["consecutive_successes"] = 1 if defense_success else -1

        # 判断是否退出
        terminated = False
        if (
            abs(self.evaluation["consecutive_successes"]) >= 10
            or self.cur_step >= self.max_steps
        ):
            terminated = True

        # 回合成功奖励
        ep_success_reward = 0
        if terminated:
            # if abs(self.evaluation["consecutive_successes"]) >= 10:
            if self.evaluation["consecutive_successes"] >= 10:
                ep_success_reward = 200
                ep_success_reward -= (self.cur_step - 10) * 5
                # ep_success_reward = 400
                # ep_success_reward -= (self.cur_step - 10) * 10
            else:
                # ep_success_reward = -320
                ep_success_reward = -200

        reward = 1 * (
            danger_penalty
            + fail_rate_penalty
            + ep_success_reward
            # + more_safe_reward
            # + delay_penalty
            + replica_cost
            # + scale_cost
            + time_cost
            # + execute_penalty
            + safety_reward
            # + replica_reward
        )

        # info
        info = {
            "reward_detail": {
                "danger_penalty": danger_penalty,
                # "more_safe_reward": more_safe_reward,
                "fail_rate_penalty": fail_rate_penalty,
                "ep_success_reward": ep_success_reward,
                # "delay_penalty": delay_penalty,
                "replica_cost": replica_cost,
                # "scale_cost": scale_cost,
                "time_cost": time_cost,
                # "execute_penalty": execute_penalty,
                "safety_reward": safety_reward,
                # "replica_reward": replica_reward,
            },
            "execute_penalties": self.execute_penalties,
            "scale_nums": self.scale_nums,
            "scale_info": scale_info,
            "defense_success": defense_success,
            "evaluation": self.evaluation,
        }

        if self.log_console:
            print("\n>>>>Info>>>>")
            print(info)

        return self.state.copy(), reward, terminated, info
