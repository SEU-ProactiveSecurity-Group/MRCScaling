import igraph as ig
import json
import random

# 文件路径
graphpath = "./social-network-workmodel.json"
outpath = "./social-network-workmodel.pdf"
selectedpath = "./selected-routes.json"
select_num = 6


def draw_directed_graph():  # 加载图结构
    with open(graphpath, "r", encoding="utf-8") as f:
        workmodel = json.load(f)

    exported_services = workmodel["s0"]["external_services"][0]["services"]
    selected_routes = random.sample(exported_services, select_num)
    selected_routes = ["s1", "s9", "s10", "s12"]

    with open(selectedpath, "w", encoding="utf-8") as f:
        json.dump(selected_routes, f, indent=2)

    # 创建有向图
    G = ig.Graph(directed=True)

    # 提取所有非 s0 的节点
    nodes = [node for node in workmodel.keys() if node != "s0"]
    G.add_vertices(nodes)

    # 设置节点属性（label）
    labels = []
    for node in nodes:
        data = workmodel[node]
        label = node
        loader = data.get("internal_service", {}).get("loader", {})
        for stress, detail in loader.items():
            if isinstance(detail, dict) and detail.get("run", False):
                if stress == "cpu_stress":
                    label += "-cpu"
                elif stress == "memory_stress":
                    label += "-mem"
                elif stress == "disk_stress":
                    label += "-disk"
                elif stress == "sleep_stress":
                    label += "-sleep"
                break
        labels.append(label)
    G.vs["label"] = labels
    G.vs["name"] = nodes  # 确保 name 属性存在

    # 添加边和概率属性
    edges = []
    probabilities = []

    for node in nodes:
        data = workmodel[node]
        ext_services_list = data.get("external_services", [])
        for ext_service in ext_services_list:
            items = ext_service.get("services", [])
            probs = ext_service.get("probabilities", {})

            # probs 可能是 dict 或 list
            if isinstance(probs, dict):
                for item in items:
                    prob = float(probs.get(item, 1.0))  # 默认值为 1.0
                    if item in nodes:
                        edges.append((node, item))
                        probabilities.append(prob)
            elif isinstance(probs, list):
                for item, prob in zip(items, probs):
                    if item in nodes:
                        edges.append((node, item))
                        probabilities.append(float(prob))

    G.add_edges(edges)
    G.es["probability"] = probabilities
    G.es["curved"] = False

    # 布局
    layout = G.layout("fr")  # Fruchterman-Reingold 布局

    # 绘图样式
    visual_style = {
        "vertex_size": 30,
        "vertex_label": G.vs["label"],
        "vertex_color": [
            "red" if v["name"] in selected_routes else "lightblue" for v in G.vs
        ],
        "vertex_label_dist": 1.5,
        "vertex_label_size": 12,
        "edge_width": [1 + 4 * p for p in G.es["probability"]],  # 边宽与概率成正比
        "edge_color": "gray",
        "edge_arrow_size": 1.0,
        "edge_arrow_width": 1.5,
        # "edge_label": [f"{p:.3f}" for p in G.es["probability"]],
        "layout": layout,
        "bbox": (1000, 800),
        "margin": 50,
        "vertex_label_angle": 0,
        "vertex_label_color": [
            "purple" if "-" in v["label"] else "black" for v in G.vs
        ],
        "edge_curved": False,
    }

    # 保存图像
    ig.plot(G, outpath, **visual_style)
    print(f"✅ 有向图已成功绘制并保存至: {outpath}")


draw_directed_graph()
