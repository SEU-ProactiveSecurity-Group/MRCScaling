import random


class Greedy:
    def __init__(self):
        self.action_index = 0
        # self.scale_list = [[i, j] for i in range(0, 10) for j in range(0, 10)]
        self.scale_list = [[7, 5], [9, 5], [2, 4], [5, 5], [5, 7], [0, 1], [1, 0]]
        # self.scale_list = [[7, 5], [9, 5], [2, 4], [5, 7]]
        self.success_num = [0] * len(self.scale_list)

    def take_action(self, state):
        # 如果有超过阈值的服务，则增加副本，否则尝试缩容

        # up_threshold = 7
        # down_threshold = 5
        # up_threshold = 3
        # down_threshold = 4
        # up_threshold = 8
        # down_threshold = 6

        # self.action_index = random.choice(range(len(scale_list)))
        # 按照 success_num 计算选择概率
        total_success = sum(self.success_num)
        if total_success == 0:
            self.action_index = random.choice(range(len(self.scale_list)))
        else:
            probabilities = [num / total_success for num in self.success_num]
            self.action_index = random.choices(
                range(len(self.scale_list)), weights=probabilities, k=1
            )[0]
        [up_threshold, down_threshold] = self.scale_list[self.action_index]

        # scale_up = False
        # for service_idx, metrics in enumerate(state):
        #     if metrics[0] > up_threshold or metrics[1] > up_threshold:
        #         scale_up = True
        #         break
        # # 增加副本
        # if scale_up:
        #     threshold_scale_action = target_scale_action = up_threshold
        # else:
        #     # 尝试缩容
        #     threshold_scale_action = down_threshold - 1
        #     target_scale_action = down_threshold
        # return [threshold_scale_action, target_scale_action]

        return [up_threshold, down_threshold]

    def update(self, defense_success):
        if defense_success:
            self.success_num[self.action_index] += 1

    def reset(self):
        self.success_num = [0] * len(self.scale_list)
        self.action_index = 0
