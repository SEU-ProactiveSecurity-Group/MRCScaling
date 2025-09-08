import random


class HPA:
    def __init__(self):
        pass

    def take_action(self, state):
        # 如果有超过阈值的服务，则增加副本，否则尝试缩容

        # up_threshold = 7
        # down_threshold = 5
        # up_threshold = 3
        # down_threshold = 4
        # up_threshold = 8
        # down_threshold = 6

        scale_list = [[7, 5], [9, 5], [2, 4], [5, 7]]
        [up_threshold, down_threshold] = random.choice(scale_list)

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
