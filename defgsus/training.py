import dataclasses
import hashlib
import json
import math
import os
from io import StringIO
from pathlib import Path
from typing import Optional, List, TextIO
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
import github_slugger

import ExpToolKit


EXPERIMENTS_PATH = Path(__file__).resolve().parent / "experiments"


@dataclasses.dataclass
class Experiment:
    model: str
    train_config: Optional[dict] = None
    model_config: Optional[dict] = None
    num_trials: int = 10

    @property
    def config_file(self) -> str:
        if self.model == "MLP":
            return "model_config/config0.yaml"
        elif self.model == "KAE":
            return "model_config/config6.yaml"
        else:
            raise ValueError(f"Unknown model '{self.model}'")

    @property
    def hash(self) -> str:
        string = f"{self.model} {self.num_trials} {self.train_config} {self.model_config}"
        return hashlib.md5(string.encode()).hexdigest()

    @property
    def output_path(self) -> Path:
        return EXPERIMENTS_PATH / self.hash

    @property
    def model_name(self) -> str:
        hid = (self.model_config or {}).get("hidden_dims")
        if hid:
            hid = ",".join(str(h) for h in hid)
        if self.model == "MLP":
            if not hid:
                return self.model
            return f"{self.model} (hid={hid})"
        else:
            order = (self.model_config or {}).get("order", 3)
            if not hid:
                return f"{self.model} (p={order})"
            return f"{self.model} (hid={hid}, p={order})"

    @property
    def overrides(self) -> Optional[dict]:
        overrides = {}
        if self.model_config:
            overrides["MODEL"] = self.model_config
        if self.train_config:
            overrides["TRAIN"] = self.train_config
        return overrides or None

    @property
    def activation(self) -> List[str]:
        """Just for display purposes"""
        act = (self.model_config or {}).get("activation", ["relu", "sigmoid"])
        if not isinstance(act, (list, tuple)):
            act = [act]
        act = {a: None for a in act}
        return list(act.keys())

    @property
    def heading(self) -> str:
        act = "/".join(self.activation)

        optim = (self.train_config or {}).get("optim_type", "ADAM")
        lr = (self.train_config or {}).get("lr", 0.0001)
        bs = (self.train_config or {}).get("batch_size", 256)

        return (
            f"{self.model_name}, {act}, {optim}, lr={lr}, batch size={bs}"
        )

    @property
    def slug(self) -> str:
        return github_slugger.GithubSlugger().slug(self.heading)


def train_experiment(
        config_path: str,
        num_trials: int = 10,
        is_print: bool = False,
        overrides: Optional[dict] = None,
        out_file: Optional[TextIO] = None,
):
    print(config_path, file=out_file)

    config = ExpToolKit.load_config(config_path)
    if overrides:
        for key, value in overrides.items():
            config[key].update(value)
        print("\nupdated config:", file=out_file)
        pprint(config, stream=out_file)
    else:
        print("\nconfig:", file=out_file)
        pprint(config, stream=out_file)

    print(file=out_file)

    test_losses = []
    train_times = []
    all_train_loss_epoch = []
    all_test_loss_epoch = []
    models = []
    with tqdm(total=num_trials, desc="trial") as progress:
        for trial in range(num_trials):
            torch.cuda.empty_cache()

            config["TRAIN"]["random_seed"] = 2024 + trial
            train_setting = ExpToolKit.create_train_setting(config)
            if trial == 0:
                pprint(train_setting, stream=out_file)
                num_params = sum(
                    math.prod(p.shape)
                    for p in train_setting["model"].parameters()
                    if p.requires_grad
                )
                print(f"\nmodel parameters: {num_params:,}\n", file=out_file)

            print(f"trial {trial + 1}/{num_trials}", file=out_file)
            model, train_loss_epoch, train_loss_batch, epoch_time, test_loss_epoch \
                = ExpToolKit.train_and_test(**train_setting, is_print=is_print)

            print(f"test loss: {test_loss_epoch[-1]}, seconds: {epoch_time[-1]}", file=out_file)
            models.append(model)
            test_losses.append(test_loss_epoch[-1])
            train_times.append(epoch_time[-1])
            all_train_loss_epoch.append(train_loss_epoch)
            all_test_loss_epoch.append(test_loss_epoch)

            progress.set_postfix({"test loss": test_loss_epoch[-1]})
            progress.update()

    print(f"\naverage/best test loss: {sum(test_losses)/len(test_losses)} / {min(test_losses)}", file=out_file)

    return {
        "config": config,
        "model_repr": repr(model),
        "num_parameters": num_params,
        "models": models,
        "test_losses": test_losses,
        "train_times": train_times,
        "all_train_loss_epoch": all_train_loss_epoch,
        "all_test_loss_epoch": all_test_loss_epoch,
    }


def train_experiments(experiments: List[Experiment], reset: bool = False):
    for experiment in experiments:
        output_path = experiment.output_path

        if not reset and (output_path / "result.json").exists():
            continue

        print(f"\n--- Running {experiment} ---")

        string_io = StringIO()

        result = train_experiment(
            config_path=experiment.config_file,
            num_trials=experiment.num_trials,
            overrides=experiment.overrides,
            out_file=string_io,
        )
        string_io.seek(0)
        text_output = string_io.read()

        best_index = int(np.array(result["test_losses"]).argmin())
        best_model = result.pop("models")[best_index]

        os.makedirs(output_path, exist_ok=True)
        torch.save(best_model, output_path / "model.pt")
        (output_path / "result.json").write_text(json.dumps({
            **result,
            "text_output": text_output,
        }, indent=2))
