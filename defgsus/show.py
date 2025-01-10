import json
import os
from pathlib import Path
from typing import Optional, List, TextIO

import torch
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
import torchvision.transforms.functional as VF
from tqdm import tqdm

import ExpToolKit

from .training import Experiment


def show_experiments_table(
        experiments: List[Experiment],
        slugs: bool = False,
        tests: bool = False,
        markdown: bool = False,
):
    rows = []
    for experiment in experiments:
        result_file = experiment.output_path / "result.json"
        test_result_file = experiment.output_path / "test-results.json"

        if not result_file.exists():
            continue

        result = json.loads(result_file.read_text())

        test_result = None
        if tests and test_result_file.exists():
            test_result = json.loads(test_result_file.read_text())

        losses = np.array(result["test_losses"])
        times = np.array(result["train_times"])
        epochs = result["config"]["TRAIN"]["epochs"]

        model_name = experiment.model_name
        if slugs:
            model_name = f"[{model_name}](#{experiment.slug})"

        if markdown:
            nbsp = "&nbsp;"
            br = "<br/>"
            test_loss = f"""{round(losses.mean(), 4)}{nbsp}<span class="small">±{round(losses.std(), 4)}</span>"""
        else:
            nbsp, br = " ", " "
            test_loss = f"""{round(losses.mean(), 4)} ±{round(losses.std(), 4)}"""

        row = {
            "model": model_name,
            "act": "/".join(experiment.activation),
            "params": "{num_parameters:,}".format(**result),
        }
        if not test_result:
            row.update({
                "optim": result["config"]["TRAIN"]["optim_type"].capitalize(),
                "lr": result["config"]["TRAIN"]["lr"],
                "batch size": result["config"]["TRAIN"]["batch_size"],
            })
        else:
            row.update({
                "optim/lr/bs": "{}/{}/{}".format(
                    result["config"]["TRAIN"]["optim_type"].capitalize(),
                    result["config"]["TRAIN"]["lr"],
                    result["config"]["TRAIN"]["batch_size"],
                )
            })
        row.update({
            f"test loss{br}({losses.shape[0]} runs)↓": test_loss,
            f"train time{br}({epochs} ep)": f"{round(times.mean(), 1)} sec",
        })
        if test_result:
            row.update({
                f"classifier{br}accuracy↑": round(test_result["classifier"]["test"], 4),
                f"retriever{br}recall@5↑": round(test_result["retriever"]["test"], 4),
                f"denoiser{br}salt&pepper↓": round(test_result["denoiser_sp"]["test"], 4),
            })
        rows.append(row)

    df = pd.DataFrame(rows)

    print(df.to_markdown(
        index=False,
        colalign=["left", "left", "right", "left", "right", "right", "right", "right"]
    ))


def show_test_table(experiments: List[Experiment]):
    rows = []
    for experiment in experiments:
        result_file = experiment.output_path / "result.json"
        test_result_file = experiment.output_path / "test-results.json"

        if not (result_file.exists() and test_result_file.exists()):
            continue

        result = json.loads(result_file.read_text())
        test_result = json.loads(test_result_file.read_text())

        losses = np.array(result["test_losses"])
        times = np.array(result["train_times"])
        epochs = result["config"]["TRAIN"]["epochs"]

        rows.append({
            "model": experiment.model_name,
            "activation": "/".join(experiment.activation),
            "params": "{num_parameters:,}".format(**result),
            "optim": result["config"]["TRAIN"]["optim_type"],
            "learnrate": result["config"]["TRAIN"]["lr"],
            "batch size": result["config"]["TRAIN"]["batch_size"],
            f"classifier": round(test_result["classifier"]["test"], 4),
            f"retriever": round(test_result["retriever"]["test"], 4),
            f"denoiser": round(test_result["denoiser_sp"]["test"], 4),
        })

    df = pd.DataFrame(rows)

    print(df.to_markdown(
        index=False,
        colalign=["left", "left", "right", "left", "right", "right", "right", "right", "right"]
    ))


def show_markdown(experiments: List[Experiment]):
    for experiment in experiments:
        result_file = experiment.output_path / "result.json"

        if not result_file.exists():
            continue

        result = json.loads(result_file.read_text())

        print(f"\n\n### {experiment.heading}\n")

        print("```")
        print(result["text_output"])
        print("```")


def list_experiments(experiments: List[Experiment]):
    for experiment in experiments:
        print(f"{experiment.hash} {experiment}")


def render_reconstructions(experiments: List[Experiment]):
    for experiment in tqdm(experiments, desc="render reconstructions"):
        out_file = Path(__file__).resolve().parent / "images" / f"reconstruction-{experiment.slug}.png"
        if out_file.exists():
            continue

        model_file = experiment.output_path / "model.pt"
        if not model_file.exists():
            continue

        config = ExpToolKit.load_config(experiment.config_file)
        if experiment.overrides:
            for key, value in experiment.overrides.items():
                config[key].update(value)

        setting = ExpToolKit.create_train_setting(config)

        model = torch.load(model_file)
        model.eval()

        batch = None
        for batch0 in setting["test_loader"]:
            batch0 = batch0[0].to(setting["device"])
            if batch is None:
                batch = batch0
            else:
                batch = torch.cat([batch, batch0])
            if batch.shape[0] > 64:
                break
        batch = batch[:64]

        out_batch = model(batch).clamp(0, 1).reshape(batch.shape)
        grid = []
        for i in range(8):
            grid2 = []
            for j in range(8):
                grid2.append(batch[i * 8 + j])
                grid2.append(out_batch[i * 8 + j])
            grid.append(make_grid(grid2, nrow=2))

        image = VF.to_pil_image(make_grid(grid, nrow=8, padding=3, pad_value=.3))

        os.makedirs(out_file.parent, exist_ok=True)

        image.save(out_file)
