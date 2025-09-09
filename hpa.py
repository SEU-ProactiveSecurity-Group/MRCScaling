import random


class HPA:
    def __init__(self):
        pass

    def take_action(self, state):
        # 如果有超过阈值的服务，则增加副本，否则尝试缩容
        scale_list = [[7, 5], [9, 5], [2, 4], [5, 7]]
        [up_threshold, down_threshold] = random.choice(scale_list)

        return [up_threshold, down_threshold]
