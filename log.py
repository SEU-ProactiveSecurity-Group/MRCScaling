import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import json


class Logger:
    def __init__(
        self,
        title: str,
        log_dir: str = "log",
        csv_dir: str = "csv",
        json_dir: str = "json",
    ):
        self.dir_path = f"{os.getcwd()}/output"
        # self.timestamp = f
        self.log_path = f"{self.dir_path}/{title}-{title}/{log_dir}"
        self.csv_path = f"{self.dir_path}/{title}-{title}/{csv_dir}"
        self.json_path = f"{self.dir_path}/{title}-{title}/{json_dir}"

        self.log_writers: dict[str, SummaryWriter] = {}
        self.csv_writers: dict[str, pd.DataFrame] = {}
        self.json_writers: dict[str, list] = {}

    def close(self):
        # flush 所有日志再 close
        for writer in self.log_writers.values():
            writer.flush()
            writer.close()
        for name, df in self.csv_writers.items():
            df.to_csv(os.path.join(self.csv_path, f"{name}.csv"), index=False)
        for name, data in self.json_writers.items():
            with open(
                os.path.join(self.json_path, f"{name}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # tensorboard log
    def write_logs(self, x: int, logs: dict[str, float | int]):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        for name, y in logs.items():
            self.write_log(name, x, y)

    def write_log(
        self,
        name: str,
        x: int,
        y: float | int,
    ):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if name not in self.log_writers:
            self.log_writers[name] = SummaryWriter(f"{self.log_path}")
        self.log_writers[name].add_scalar(name, y, x)

    # csv log
    def write_csv(self, name: str, episode: int, data: dict[str, float | int]):
        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)
        file_path = os.path.join(self.csv_path, f"{name}.csv")
        if name not in self.csv_writers:
            columns = list(data.keys())
            self.csv_writers[name] = pd.DataFrame(columns=columns)
        self.csv_writers[name].loc[episode] = data.values()
        if episode % 10 == 0:
            self.csv_writers[name].to_csv(file_path, index=False)

    # json log
    def write_json(self, name: str, episode: int, data):
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)
        file_path = os.path.join(self.json_path, f"{name}.json")
        if name not in self.json_writers:
            self.json_writers[name] = list()
        self.json_writers[name].append(data)
        if episode % 10 == 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.json_writers[name], ensure_ascii=False, indent=2)
                )
