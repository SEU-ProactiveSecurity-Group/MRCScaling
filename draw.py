import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

parser = argparse.ArgumentParser(description="argparse")
parser.add_argument(
    "--jsonname", type=str, required=True, help="Name of the JSON file to analyze"
)
args = parser.parse_args()
jsonname = args.jsonname

jsonpath = f"./output/{jsonname}/json/info.json"

output_dir = f"./graph/{jsonname}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(jsonpath, "r", encoding="utf-8") as f:
    data = json.load(f)

test_data = data[-10:]

# 1. 平均步数
average_steps = sum([len(ep_data) for ep_data in test_data]) / len(test_data)
print(f"平均步数: {average_steps}")

# 2.1 每步成功率
success_count = 0
total_steps = 0
for ep_data in test_data:
    for item in ep_data:
        if item["defense_success"]:
            success_count += 1
        total_steps += 1
success_rate = success_count / total_steps if total_steps > 0 else 0
print(f"每步成功率: {success_rate:.2%}")


# 2.2 回合成功率
ep_success_count = 0
for ep_data in test_data:
    if any(not item["defense_success"] for item in ep_data[-10:]):
        continue
    ep_success_count += 1
ep_success_rate = ep_success_count / len(test_data) if len(test_data) > 0 else 0
print(f"回合成功率: {ep_success_rate:.2%}")

# 3. 危险节点数随步数变化
def draw_sns_line(data, x, y, title, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x, y=y, estimator="mean", ci="sd", marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


dangerous_nodes = pd.DataFrame(columns=range(0, 20))
for ep, ep_data in enumerate(test_data):
    for step, item in enumerate(ep_data):
        dangerous_nodes.loc[ep, step] = item["evaluation"]["dangerous_nodes"]
# 重新整理数据格式以适合seaborn
plot_data = []
for ep in range(len(dangerous_nodes)):
    for step in dangerous_nodes.columns:
        if not pd.isna(dangerous_nodes.loc[ep, step]):
            plot_data.append(
                {
                    "step": step,
                    "dangerous_nodes": dangerous_nodes.loc[ep, step],
                    "episode": ep,
                }
            )

plot_df = pd.DataFrame(plot_data)
# 用seaborn画带有误差条的折线图
draw_sns_line(
    plot_df,
    "step",
    "dangerous_nodes",
    "Dangerous Nodes Over Steps",
    f"{output_dir}/dangerous_nodes_over_steps.png",
)

# 4. 副本数随时间变化
replica_counts = pd.DataFrame(columns=range(0, 20))
for ep, ep_data in enumerate(test_data):
    for step, item in enumerate(ep_data):
        replica_counts.loc[ep, step] = item["evaluation"]["num_replicas"]
# 重新整理数据格式以适合seaborn
plot_data = []
for ep in range(len(replica_counts)):
    for step in replica_counts.columns:
        if not pd.isna(replica_counts.loc[ep, step]):
            plot_data.append(
                {
                    "step": step,
                    "num_replicas": replica_counts.loc[ep, step],
                    "episode": ep,
                }
            )
plot_df = pd.DataFrame(plot_data)
# 用seaborn画带有误差条的折线图
draw_sns_line(
    plot_df,
    "step",
    "num_replicas",
    "Number of Replicas Over Steps",
    f"{output_dir}/num_replicas_over_steps.png",
)

# 5. 空闲副本数随时间变化
idle_replica_counts = pd.DataFrame(columns=range(0, 20))
for ep, ep_data in enumerate(test_data):
    for step, item in enumerate(ep_data):
        idle_replica_counts.loc[ep, step] = sum(
            item["evaluation"]["unused_replicas"].values()
        )
# 重新整理数据格式以适合seaborn
plot_data = []
for ep in range(len(idle_replica_counts)):
    for step in idle_replica_counts.columns:
        if not pd.isna(idle_replica_counts.loc[ep, step]):
            plot_data.append(
                {
                    "step": step,
                    "idle_replicas": idle_replica_counts.loc[ep, step],
                    "episode": ep,
                }
            )
plot_df = pd.DataFrame(plot_data)
# 用seaborn画带有误差条的折线图
draw_sns_line(
    plot_df,
    "step",
    "idle_replicas",
    "Idle Replicas Over Steps",
    f"{output_dir}/idle_replicas_over_steps.png",
)


# 6. 各个链路时延随时间变化，draw_sns_line画图同时，对于每条链路时延画一条线，所以一张图上有多条带阴影的折线
def draw_sns_line_multi(data, x, y, hue, title, save_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x=x, y=y, hue=hue, estimator="mean", ci="sd", marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


link_delays = {}
for ep, ep_data in enumerate(test_data):
    for step, item in enumerate(ep_data):
        for link, delay in item["evaluation"]["route_delay"].items():
            if link not in link_delays:
                link_delays[link] = pd.DataFrame(columns=range(0, 20))
            link_delays[link].loc[ep, step] = delay
# 重新整理数据格式以适合seaborn
plot_data = []
for link, delays in link_delays.items():
    for ep in range(len(delays)):
        for step in delays.columns:
            if not pd.isna(delays.loc[ep, step]):
                plot_data.append(
                    {
                        "step": step,
                        "delay": delays.loc[ep, step],
                        "episode": ep,
                        "link": link,
                    }
                )
plot_df = pd.DataFrame(plot_data)
# 用seaborn画带有误差条的折线图
draw_sns_line_multi(
    plot_df,
    "step",
    "delay",
    "link",
    "Link Delays Over Steps",
    f"{output_dir}/link_delays_over_steps.png",
)

# 7. 各个路由失败率随步数变化
route_fail_rates = {}
for ep, ep_data in enumerate(test_data):
    for step, item in enumerate(ep_data):
        for route, fail_rate in item["evaluation"]["route_fail_rate"].items():
            if route not in route_fail_rates:
                route_fail_rates[route] = pd.DataFrame(columns=range(0, 20))
            route_fail_rates[route].loc[ep, step] = fail_rate
# 重新整理数据格式以适合seaborn
plot_data = []
for route, fail_rates in route_fail_rates.items():
    for ep in range(len(fail_rates)):
        for step in fail_rates.columns:
            if not pd.isna(fail_rates.loc[ep, step]):
                plot_data.append(
                    {
                        "step": step,
                        "fail_rate": fail_rates.loc[ep, step],
                        "episode": ep,
                        "route": route,
                    }
                )
plot_df = pd.DataFrame(plot_data)
# 用seaborn画带有误差条的折线图
draw_sns_line_multi(
    plot_df,
    "step",
    "fail_rate",
    "route",
    "Route Fail Rates Over Steps",
    f"{output_dir}/route_fail_rates_over_steps.png",
)
