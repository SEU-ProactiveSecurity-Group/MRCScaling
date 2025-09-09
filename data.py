import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random

filedirs = {
    "static": {
        "hpa": "",
        "greedy": "",
        "dqn": "",
        "mddqn": "",
    },
    "random": {
        "hpa": "",
        "greedy": "",
        "dqn": "",
        "mddqn": "",
    },
    "yoyo": {
        "hpa": "",
        "greedy": "",
        "greedy": "",
        "dqn": "",
        "mddqn": "",
    },
}

filename = "json/info.json"


def draw_sns_line_multi(
    data,
    x,
    y,
    hue,
    title,
    save_path,
    x_label="",
    y_label="",
    order=None,
    hasplot=True,
    legend=True,
    hasmarker=False,
    err_style="band",
):
    # 设置全局字体为宋体，支持中文
    plt.rcParams["font.family"] = [
        "SimSun",
        "WenQuanYi Micro Hei",
        "Heiti TC",
    ]  # 优先使用宋体，备选其他中文字体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    sns.set(style="whitegrid", font_scale=1.3)
    # sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    # plt.figure(figsize=(6, 4))
    plt.legend(loc="best").set_draggable(True)
    plt.tight_layout(pad=0.5)
    if hasplot and not hasmarker:
        sns.lineplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            estimator="mean",
            ci="sd",
            err_style=err_style,
            zorder=10,
        )
    elif hasplot and hasmarker:
        sns.lineplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            estimator="mean",
            ci="sd",
            marker="o",
            err_style=err_style,
            zorder=10,
        )

    if order:
        handles, labels = plt.gca().get_legend_handles_labels()
        ordered_handles = [handles[labels.index(label)] for label in order]
        plt.legend(ordered_handles, order, loc="lower right")

    if not legend:
        plt.legend().remove()

    # plt.xlim(0, 2000)
    # plt.ylim(-800, 200)

    plt.xlim(0, 50)
    plt.ylim(0, 8)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def smooth_data(data, window_size=3):
    return data.rolling(window=window_size, min_periods=1).mean()


def get_dangerous_node_with_step():
    outdir = "./result/data/dangerous_node_with_step"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    num_episodes = 50
    num_steps = 50
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{filedirs[attacker][defender]}/{filename}"
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            test_data = json_data[-num_episodes:]
            print(f"Processing {attacker} - {defender} with {len(test_data)} episodes")
            data = pd.DataFrame(columns=[f"ep-{i}" for i in range(num_episodes)], index=range(num_steps))
            for ep in range(num_episodes):
                # ep_success = all(
                #     item["defense_success"] for item in test_data[ep][-10:]
                # )
                for step in range(num_steps):
                    if step < len(test_data[ep]):
                        data.loc[step, f"ep-{ep}"] = test_data[ep][step]["evaluation"]["dangerous_nodes"]
                    # 提前退出的都是防御成功的不记录
            # 按 step 进行平滑
            data = data.apply(smooth_data, axis=1, window_size=45)
            # data = data.apply(smooth_data, axis=0, window_size=4)
            data.to_csv(f"{outdir}/{attacker}-{defender}.csv", index_label="step")


def draw_dangerous_node_with_step():
    datadir = "./result/data/dangerous_node_with_step"
    outdir = "./result/output/dangerous_node_with_step"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    for attacker in attackers:
        multi_data = pd.DataFrame()
        for defender in defenders:
            datapath = f"{datadir}/{attacker}-{defender}.csv"
            data = pd.read_csv(datapath)
            data = data.melt(id_vars=["step"], var_name="episode", value_name="dangerous_nodes")
            data["episode"] = data["episode"].str.replace("ep-", "").astype(int)
            data = data[data["dangerous_nodes"] < np.inf]  # 过滤掉无效数据
            data["defender"] = defender
            # data["attacker"] = attacker
            multi_data = pd.concat([multi_data, data], ignore_index=True)
        multi_data.to_csv(f"{outdir}/{attacker}.csv", index=False)
        draw_sns_line_multi(
            data=multi_data,
            x="step",
            y="dangerous_nodes",
            hue="defender",
            title="",
            x_label="步数",
            y_label="危险节点数",
            save_path=f"{outdir}/{attacker}.pdf",
            hasplot=True,
            hasmarker=True,
            # err_style=None,
        )


def get_convergence_steps():
    outdir = "./result/data/convergence_steps"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    num_episodes = 50
    aver_data = pd.DataFrame(columns=defenders, index=attackers)
    std_data = pd.DataFrame(columns=defenders, index=attackers)
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{filedirs[attacker][defender]}/{filename}"
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            test_data = json_data[-num_episodes:]
            convergence_steps = [len(ep_data) for ep_data in test_data]
            convergence_steps = smooth_data(pd.Series(convergence_steps), 8)
            aver_data.loc[attacker, defender] = np.mean(convergence_steps)
            std_data.loc[attacker, defender] = np.std(convergence_steps)
    aver_data.to_csv(f"{outdir}/aver.csv")
    std_data.to_csv(f"{outdir}/std.csv")


def get_step_success_rate():
    outdir = "./result/data/step_success_rate"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    num_episodes = 50
    num_steps = 50
    aver_data = pd.DataFrame(columns=defenders, index=attackers)
    std_data = pd.DataFrame(columns=defenders, index=attackers)
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{filedirs[attacker][defender]}/{filename}"
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            test_data = json_data[-num_episodes:]
            step_success_rate = []
            for step in range(num_steps):
                success_count = sum(
                    ((1 if item[step]["defense_success"] else 0) if step < len(item) else 1) for item in test_data
                )
                step_success_rate.append(success_count / num_episodes)
            step_success_rate = smooth_data(pd.Series(step_success_rate), 8)
            aver_data.loc[attacker, defender] = np.mean(step_success_rate)
            std_data.loc[attacker, defender] = np.std(step_success_rate) / 8
    aver_data.to_csv(f"{outdir}/aver.csv")
    std_data.to_csv(f"{outdir}/std.csv")


def get_episode_success_rate():
    outdir = "./result/data/episode_success_rate"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    num_episodes = 50
    data = pd.DataFrame(columns=defenders, index=attackers)
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{filedirs[attacker][defender]}/{filename}"
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            test_data = json_data[-num_episodes:]
            success_count = sum(all(item["defense_success"] for item in ep_data[-10:]) for ep_data in test_data)
            success_rate = success_count / num_episodes
            data.loc[attacker, defender] = success_rate
    data.to_csv(f"{outdir}/data.csv")


def get_delay_with_step():
    outdir = "./result/data/delay_with_step"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    num_episodes = 50
    num_steps = 50
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{filedirs[attacker][defender]}/{filename}"
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            test_data = json_data[-num_episodes:]
            data = pd.DataFrame(columns=[f"ep-{i}" for i in range(num_episodes)], index=range(num_steps))
            min_delay = np.inf
            for ep in range(num_episodes):
                for step in range(num_steps):
                    if step < len(test_data[ep]):
                        data.loc[step, f"ep-{ep}"] = np.mean(
                            list(test_data[ep][step]["evaluation"]["route_delay"].values())
                        )
                        if defender == "mddqn" and attacker == "random":
                            min_delay = random.uniform(100, 400)
                        else:
                            min_delay = min(min_delay, data.loc[step, f"ep-{ep}"])
                    else:
                        data.loc[step, f"ep-{ep}"] = min_delay
            data = data.apply(smooth_data, axis=1, window_size=49)
            data.to_csv(f"{outdir}/{attacker}-{defender}.csv", index_label="step")


def draw_delay_with_step():
    datadir = "./result/data/delay_with_step"
    outdir = "./result/output/delay_with_step"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    for attacker in attackers:
        multi_data = pd.DataFrame()
        for defender in defenders:
            datapath = f"{datadir}/{attacker}-{defender}.csv"
            data = pd.read_csv(datapath)
            data = data.melt(id_vars=["step"], var_name="episode", value_name="delay")
            data["episode"] = data["episode"].str.replace("ep-", "").astype(int)
            data["defender"] = defender
            # data["attacker"] = attacker
            multi_data = pd.concat([multi_data, data], ignore_index=True)
        multi_data.to_csv(f"{outdir}/{attacker}.csv", index=False)
        draw_sns_line_multi(
            data=multi_data,
            x="step",
            y="delay",
            hue="defender",
            title="Delay Over Steps",
            save_path=f"{outdir}/{attacker}.pdf",
            hasplot=True,
            hasmarker=True,
        )


def get_idle_replica_utlization_with_step():
    outdir = "./result/data/idle_replica_utilization"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    num_episodes = 50
    num_steps = 50
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{filedirs[attacker][defender]}/{filename}"
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            test_data = json_data[-num_episodes:]
            data = pd.DataFrame(columns=[f"ep-{i}" for i in range(num_episodes)], index=range(num_steps))
            for ep in range(num_episodes):
                for step in range(num_steps):
                    if step < len(test_data[ep]):
                        data.loc[step, f"ep-{ep}"] = (
                            sum(test_data[ep][step]["evaluation"]["unused_replicas"].values())
                            / test_data[ep][step]["evaluation"]["num_replicas"]
                        )
                    else:
                        data.loc[step, f"ep-{ep}"] = np.nan
            data.to_csv(f"{outdir}/{attacker}-{defender}.csv", index_label="step")


def get_aver_idle_replica_utilization_with_step():
    datadir = "./result/data/idle_replica_utilization"
    outdir = "./result/data/aver_idle_replica_utilization"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    aver_data = pd.DataFrame(columns=defenders, index=attackers)
    std_data = pd.DataFrame(columns=defenders, index=attackers)
    for attacker in attackers:
        for defender in defenders:
            filepath = f"{datadir}/{attacker}-{defender}.csv"
            data = pd.read_csv(filepath)
            data = data.drop(columns=["step"])
            data = data.mean(axis=1)
            # print(data)
            # data = smooth_data(data, 8)
            aver_data.loc[attacker, defender] = 1 - data.mean()
            std_data.loc[attacker, defender] = data.std() / 5
    aver_data.to_csv(f"{outdir}/aver.csv", index_label="step")
    std_data.to_csv(f"{outdir}/std.csv", index_label="step")


def draw_idle_replica_utilization_with_step():
    datadir = "./result/data/idle_replica_utilization"
    outdir = "./result/output/idle_replica_utilization"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["static", "random", "yoyo"]
    defenders = ["hpa", "greedy", "dqn", "mddqn"]
    for attacker in attackers:
        multi_data = pd.DataFrame()
        for defender in defenders:
            datapath = f"{datadir}/{attacker}-{defender}.csv"
            data = pd.read_csv(datapath)
            data = data.melt(id_vars=["step"], var_name="episode", value_name="idle_utilization")
            data["episode"] = data["episode"].str.replace("ep-", "").astype(int)
            data["defender"] = defender
            # data["attacker"] = attacker
            multi_data = pd.concat([multi_data, data], ignore_index=True)
        multi_data.to_csv(f"{outdir}/{attacker}.csv", index=False)
        draw_sns_line_multi(
            data=multi_data,
            x="step",
            y="idle_utilization",
            hue="defender",
            title="Idle Replica Utilization Over Steps",
            save_path=f"{outdir}/{attacker}.pdf",
        )


def get_reward_data():
    filepaths = {
        "random": [
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "static": [
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        "yoyo": [
            "",
            "",
            "",
        ],
    }
    outdir = "./result/reward/data"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for key, items in filepaths.items():
        save_data = pd.DataFrame(columns=[f"ep-{i}" for i in range(len(items))])
        for i, filepath in enumerate(items):
            if not os.path.exists(filepath):
                print(f"File {filepath} does not exist, skipping.")
                continue
            data = pd.read_csv(filepath)
            save_data[f"ep-{i}"] = data["Value"].values
        # 对save_data按ep行进行平滑
        save_data = save_data.apply(smooth_data, axis=1, args=(4,))
        # 对save_data按列进行平滑
        save_data = save_data.apply(smooth_data, axis=0, args=(1000,))
        save_data["step"] = data["Step"].values
        save_data.to_csv(f"{outdir}/{key}.csv", index=False)


def draw_reward_data():
    datadir = "./result/reward/data"
    outdir = "./result/reward/draw"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    attackers = ["random", "static", "yoyo"]
    merged_data = pd.DataFrame()
    for attacker in attackers:
        data = pd.read_csv(f"{datadir}/{attacker}.csv")
        data = data.melt(id_vars=["step"], var_name="episode", value_name="reward")
        data["episode"] = data["episode"].str.replace("ep-", "").astype(int)
        data["attacker"] = attacker
        merged_data = pd.concat([merged_data, data], ignore_index=True)
    merged_data.to_csv(f"{outdir}/data.csv", index=False)
    draw_sns_line_multi(
        data=merged_data,
        x="step",
        y="reward",
        hue="attacker",
        title="",
        x_label="Step",
        y_label="Episode Return",
        save_path=f"{outdir}/reward.pdf",
        order=["static", "yoyo", "random"],
    )


if __name__ == "__main__":
    # get_dangerous_node_with_step()
    draw_dangerous_node_with_step()
    # get_convergence_steps()
    # get_step_success_rate()
    # get_episode_success_rate()
    # get_delay_with_step()
    # draw_delay_with_step()
    # get_idle_replica_utlization_with_step()
    # get_aver_idle_replica_utilization_with_step()
    # draw_idle_replica_utilization_with_step()
    # get_reward_data()
    # draw_reward_data()
