import mddqn_run
import dqn_run
import hpa_run
from env import Env
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argparse")
    parser.add_argument(
        "--attacker", type=str, required=True, help="Type of the attacker"
    )
    parser.add_argument(
        "--decider", type=str, required=True, help="Type of the decider"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2000, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_episode_step", type=int, default=50, help="Maximum steps per episode"
    )
    parser.add_argument("--title", type=str, default="demo", help="Title for the run")
    parser.add_argument(
        "--log_console", type=bool, default=False, help="Log to console"
    )
    args = parser.parse_args()
    attacker = args.attacker
    decider = args.decider
    num_episodes = args.num_episodes
    max_episode_step = args.max_episode_step
    title = args.title
    log_console = args.log_console

    if attacker not in ["static", "random", "yoyo"]:
        raise ValueError(f"Unknown attacker type: {attacker}")

    if decider == "mddqn":
        env = Env(
            multi_action_dim=True,
            attacker_type=attacker,
            max_steps=max_episode_step,
            log_console=log_console,
        )
        mddqn_run.run_and_test(env, num_episodes, title)
    elif decider == "dqn":
        env = Env(
            multi_action_dim=False,
            attacker_type=attacker,
            max_steps=max_episode_step,
            log_console=log_console,
        )
        dqn_run.run_and_test(env, num_episodes, title)
    elif decider == "hpa":
        env = Env(
            multi_action_dim=True,
            attacker_type=attacker,
            max_steps=max_episode_step,
            log_console=log_console,
        )
        hpa_run.run_and_test(env, num_episodes, title)
    elif decider == "greedy":
        from greedy_run import run_and_test
        env = Env(
            multi_action_dim=True,
            attacker_type=attacker,
            max_steps=max_episode_step,
            log_console=log_console,
        )
        run_and_test(env, num_episodes, title)
    else:
        raise ValueError(f"Unknown decider type: {decider}")
