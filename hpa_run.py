from hpa import HPA
import time
from tqdm import tqdm
from log import Logger


def run_and_test(env, num_episodes=10, title="demo"):
    agent = HPA()
    timestamp = time.strftime("%m%d-%H%M%S")
    final_title = f"{title}-hpa-{env.attacker_type}-{timestamp}"
    logger = Logger(title=final_title)
    with tqdm(total=num_episodes) as pbar:
        for episode in range(num_episodes):
            step = 0
            terminated = False

            episode_return = 0
            episode_reward_detail = {}

            state_list = []
            action_list = []
            info_list = []

            state = env.reset()

            while terminated is False:
                action = agent.take_action(state)
                next_state, reward, terminated, info = env.step(action)

                episode_return += reward
                action_list.append(action)
                state_list.append(state.tolist())
                info_list.append(info)

                state = next_state
                step += 1

            ep_return_data = {"return": episode_return, **episode_reward_detail}

            # tensorboard log
            logger.write_logs(episode, ep_return_data)

            # csv log
            logger.write_csv("return", episode, ep_return_data)

            # json log
            logger.write_json("action", episode, action_list)
            logger.write_json("state", episode, state_list)
            logger.write_json("info", episode, info_list)

            pbar.update(1)
    logger.close()
