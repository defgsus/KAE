import json
from typing import List, Optional

import torch
from tqdm import tqdm

import ClassifierPack
import RetrieverPack
import DenoiserPack
import ExpToolKit

from .training import Experiment


def test_experiment(experiment: Experiment, progress: Optional[tqdm]):
    torch.cuda.empty_cache()

    config = ExpToolKit.load_config(experiment.config_file)
    if experiment.overrides:
        for key, value in experiment.overrides.items():
            config[key].update(value)

    setting = ExpToolKit.create_train_setting(config)

    model = torch.load(experiment.output_path / "model.pt")
    model.eval()

    print(f"testing model {experiment.heading}")

    results = {}

    classifier = ClassifierPack.Classifier(
        ClassifierPack.CLASSIFIER_CONST.KNN_CLASSIFIER, model, n_neighbors=5)
    train_accuracy, test_accuracy = ExpToolKit.evaluate_classifier(
        classifier, setting["train_loader"], setting["test_loader"]
    )
    print("classifier:", train_accuracy, test_accuracy)
    results["classifier"] = {"train": train_accuracy, "test": test_accuracy}
    if progress is not None:
        progress.update()

    retriever = RetrieverPack.Retriever(RetrieverPack.RETRIEVER_CONST.NEARSET_NEIGHBOR_RETRIEVER, model)
    (
        train_recall, train_distance_matrix_x,
        train_distance_matrix_x_latent,
        test_recall,
        test_distance_matrix_x,
        test_distance_matrix_x_latent
    ) = ExpToolKit.evaluate_retriever(
        retriever, setting["train_loader"], setting["test_loader"],
        top_K=5, retrieval_N=5, label_num=200
    )
    print("retriever:", train_recall, test_recall)
    results["retriever"] = {"train": train_recall, "test": test_recall}
    if progress is not None:
        progress.update()

    denoiser = DenoiserPack.Denoiser(model)
    train_loss, test_loss = ExpToolKit.evaluate_denoiser(
        denoiser, setting["train_loader"], setting["test_loader"],
        is_print=True, is_train=False, epochs=2,
        noise_type=DenoiserPack.DENOISER_CONST.SALT_AND_PEPPER_NOISE, noise_params=(0.05, 0.95),
    )
    print("denoiser (s&p):", train_loss, test_loss)
    results["denoiser_sp"] = {"train": train_loss, "test": test_loss}
    if progress is not None:
        progress.update()

    result_file = experiment.output_path / "test-results.json"
    result_file.write_text(json.dumps(results, indent=2))


def test_experiments(experiments: List[Experiment]):
    with tqdm(total=len(experiments) * 3, desc="testing") as progress:
        for experiment in experiments:

            if not (experiment.output_path / "model.pt").exists():
                progress.update(3)
                continue

            if (experiment.output_path / "test-results.json").exists():
                progress.update(3)
                continue

            test_experiment(experiment, progress)
