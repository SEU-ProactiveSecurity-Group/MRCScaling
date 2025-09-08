import json
import math
import random


class MultiRouteAttacker:
    def __init__(self, env):
        self.env = env
        self.graph = env.graph
        self.traffic_per_route = 200
        # route list
        # self.routes = self.load_routes(self.routepath)
        # self.num_routes = 6
        # self.num_routes = 4
        # self.num_routes = 6
        # self.num_routes = 20
        self.routes = ["s1", "s9", "s10", "s12"]
        # self.routes = ["s1", "s4", "s7", "s9", "s10", "s12"]
        # self.routes = [
        #     "s56",
        #     "s81",
        #     "s62",
        #     "s8",
        #     "s22",
        #     "s31",
        #     "s96",
        #     "s45",
        #     "s93",
        #     "s92",
        #     "s20",
        #     "s24",
        #     "s71",
        #     "s13",
        #     "s18",
        #     "s5",
        #     "s90",
        #     "s28",
        #     "s32",
        #     "s44",
        # ]
        # self.routes = [
        #     "s38",
        #     "s8",
        #     "s27",
        #     "s5",
        #     "s40",
        #     "s26",
        #     "s22",
        #     "s7",
        #     "s34",
        #     "s14",
        # ]
        self.traffics = {route: self.traffic_per_route for route in self.routes}

    def load_routes(self, routepath):
        with open(routepath, "r", encoding="utf-8") as f:
            routes = json.load(f)
        return routes

    def reset(self):
        pass

    def attack(self):
        for route in self.routes:
            if self.env.attacked_routes.get(route) is None:
                self.env.attacked_routes[route] = 0
            self.env.attacked_routes[route] += self.traffics[route]
            # 发起攻击
            self.env.load_route_traffic(route, self.traffics[route])


class RandomAttacker(MultiRouteAttacker):
    def __init__(self, env):
        super().__init__(env)
        self.num_routes = 6
        self.routes = []
        self.traffics = {}
        self.rnd = 5
        self.step = 0

    def select_random_route(self):
        return random.sample(list(self.env.routes.keys()), self.num_routes)

    def reset(self):
        pass

    def attack(self):
        # 每隔一定步数随机选择攻击链路
        if self.step % self.rnd == 0:
            self.routes = self.select_random_route()
            self.traffics = {route: self.traffic_per_route for route in self.routes}
        self.step += 1
        super().attack()


class YoYoAttacker(MultiRouteAttacker):
    def __init__(self, env):
        super().__init__(env)
        self.rnd = 5
        self.step = 0

    def reset(self):
        pass

    def attack(self):
        # 每隔一定步数进行攻击，少量攻击导致微服务持续处于扩容状态浪费资源
        if self.step % self.rnd == 0:
            super().attack()
        self.step += 1
