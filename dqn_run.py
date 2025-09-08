import torch
import time
from dqn import DQN
from tqdm import tqdm
import numpy as np
from log import Logger


device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def run_and_test(env, num_episodes=2000, title="demo"):
    state_dim = env.state_dims[0] * env.state_dims[1]
    action_dim = env.action_dims[0]

    split_num = 10

    lr = 1e-3
    hidden1_dim = 1024
    hidden2_dim = 128
    gamma = 0.98
    epsilon = 0.1
    target_update = 10
    buffer_size = 100000
    minimal_size = 100
    batch_size = 64
    agent = DQN(
        state_dim,
        hidden1_dim,
        hidden2_dim,
        action_dim,
        lr,
        gamma,
        epsilon,
        target_update,
        device,
        num_episodes,
    )

    timestamp = time.strftime("%m%d-%H%M%S")
    final_title = f"{title}-dqn-{env.attacker_type}-{timestamp}"
    logger = Logger(title=final_title)

    # list[episode]
    # ep_return_list = []  # tensorboard, csv
    # ep_danger_penalty_list = []  # tensorboard, csv
    # ep_fail_rate_penalty_list = []  # tensorboard, csv
    # ep_delay_penalty_list = []  # tensorboard, csv
    # ep_replica_cost_list = []  # tensorboard, csv
    # ep_time_cost_list = []  # tensorboard, csv
    # ep_safety_reward_list = []  # tensorboard, csv

    # list[episode * step]
    # step_actor_loss_list = []  # tensorboard
    # step_critic_loss_list = []  # tensorboard

    # list[episode][step]
    # ep_action_list = []  # json
    # ep_state_list = []  # json

    # list[episode][step]{dict}
    # ep_info_list = []  # json

    ep_return_list = []
    global_step = 0
    for i in range(split_num):
        with tqdm(total=num_episodes // split_num, desc=f"Iteration {i + 1}") as pbar:
            for episode in range(num_episodes // split_num):
                step = 0
                terminated = False

                episode_return = 0
                episode_reward_detail = {}

                state_list = []
                action_list = []
                info_list = []

                transition_dict = {
                    "states": [],
                    "actions": [],
                    "next_states": [],
                    "rewards": [],
                    "terminateds": [],
                }
                state = env.reset()  # 初始化环境状态

                while terminated is False:
                    action = agent.take_action(state.reshape((1, state_dim)))
                    next_state, reward, terminated, info = env.step([action])

                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["rewards"].append(reward)
                    transition_dict["terminateds"].append(terminated)

                    actor_loss, critic_loss = agent.update(transition_dict)

                    # tensorboard log
                    logger.write_log("actor_loss", global_step, actor_loss.item())
                    if isinstance(critic_loss, torch.Tensor):
                        logger.write_log("critic_loss", global_step, critic_loss.item())

                    episode_return += reward
                    for key, value in info["reward_detail"].items():
                        if key not in episode_reward_detail:
                            episode_reward_detail[key] = 0
                        episode_reward_detail[key] += value

                    action_list.append(action)
                    state_list.append(state.tolist())
                    info_list.append(info)

                    state = next_state
                    step += 1
                    global_step += 1

                ep_return_data = {"return": episode_return, **episode_reward_detail}

                real_episode = i * (num_episodes // 10) + episode
                # tensorboard log
                logger.write_logs(real_episode, ep_return_data)

                # csv log
                logger.write_csv("return", real_episode, ep_return_data)

                # json log
                logger.write_json("action", real_episode, action_list)
                logger.write_json("state", real_episode, state_list)
                logger.write_json("info", real_episode, info_list)

                ep_return_list.append(episode_return)

                agent.update(transition_dict)

                pbar.update(1)
                pbar.set_postfix({"Episode Return": np.mean(ep_return_list[-10:])})
    logger.close()
