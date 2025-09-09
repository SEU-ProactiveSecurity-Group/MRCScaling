import math


class PdDefender:
    def __init__(self, env):
        self.env = env
        self.graph = env.graph
        self.hosts = env.hosts
        self.routes = env.routes
        self.services = env.services
        self.num_services = env.num_services

    def reset(self):
        pass

    def defend(self, action):
        scale_info = {
            "type": "no_action",
            "threshold_scale": 0,
            "target_scale": 0,
        }
        if self.env.multi_action_dim:
            # 二维动作空间，动作维度 10 * 10，第一维或第二维取0表示不采取行动
            if not action[0] == 0 and not action[1] == 0:
                self.env.scale_replicas(action[0] * 10, action[1] * 10)
                scale_info["type"] = (
                    "scale_up" if action[0] >= action[1] else "scale_down"
                )
                scale_info["threshold_scale"] = action[0] * 10
                scale_info["target_scale"] = action[1] * 10
        else:
            # 一维动作空间，动作范围 0 到 81，0 代表不采取行动
            action = action[0] - 1
            if action >= 0:
                threshold_scale = action // 9 + 1
                target_scale = action % 9 + 1
                self.env.scale_replicas(
                    threshold_scale * 10, target_scale * 10
                )
                scale_info["type"] = (
                    "scale_up" if threshold_scale >= target_scale else "scale_down"
                )
                scale_info["threshold_scale"] = threshold_scale * 10
                scale_info["target_scale"] = target_scale * 10
        return scale_info
