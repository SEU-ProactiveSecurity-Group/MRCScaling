import json

filepath = "/disk2/duk/work-disk2/pd-for-ms/output/v1-dqn-random-0729-124527-v1-dqn-random-0729-124527/json"
action_file = f"{filepath}/action.json"
state_file = f"{filepath}/state.json"
info_file = f"{filepath}/info.json"

with open(action_file, "r", encoding="utf-8") as f:
    action_data = json.load(f)

with open(state_file, "r", encoding="utf-8") as f:
    state_data = json.load(f)

with open(info_file, "r", encoding="utf-8") as f:
    info_data = json.load(f)

output_file = f"{filepath}/output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for episode, (action, state, info) in enumerate(
        zip(action_data, state_data, info_data)
    ):
        f.write(f"=============== Episode {episode} ===============\n")
        for step, (act, st, inf) in enumerate(zip(action, state, info)):
            f.write(f">>>>>>>>> Step {step} <<<<<<<<<\n")
            f.write(f"Action: {act}\n")
            f.write("State:\n")
            for item in st:
                f.write(f"  {item}\n")
            f.write("Info:\n")
            for key, value in inf.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
