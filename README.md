# KAE: Kolmogorov-Arnold Auto-Encoder

The source code for the arXiv paper titled [**"KAE: Kolmogorov-Arnold Auto-Encoder for Representation Learning"**](https://arxiv.org/pdf/2501.00420) is available. This code, created by [Ruilizhen Hu](https://github.com/HuRuilizhen), [Fangchen Yu](https://github.com/SciYu), and [Yidong Lin](https://github.com/Asuna-L), supports experiments with Auto-Encoders on various tasks, utilizing different blocks and architectures.

## Code Structure

The project consists of four main packages: `DenseLayerPack`, `ClassifierPack`, `RetrieverPack`, `DenoiserPack`, and an additional utility package, `ExpToolKit`.

### DenseLayerPack

The `DenseLayerPack` contains implementations of various dense layers, including `DenseLayer`, `KAN`, `FourierKAN`, `WavKAN`, and the proposed `KAE`. Detailed information can be found in the `DenseLayerPack` folder.

This package is built upon the foundational work of [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/), [FourierKAN](https://github.com/GistNoesis/FourierKAN/), [WavKAN](https://github.com/zavareh1/Wav-KAN), [Efficient-KAN](https://github.com/Blealtan/efficient-kan), and [KAN](https://github.com/KindXiaoming/pykan). Leveraging these contributions, we have created the `DenseLayerPack`, which allows seamless implementation of `KAE`, `KAN`, `FourierKAN`, `WavKAN`, and `Linear` layers using a unified PyTorch interface. 

*Please note that the copyright for the original code belongs to the respective authors mentioned above.*

Available methods or classes are as follows:

- DenseLayerPack.DenseLayer: `DenseLayer(in_features: int, out_features: int, layer_type: str, **kwargs)` Initialize a specific type of layer based on the layer_type argument and additional arguments.

- DenseLayerPack.DENSE_LAYER_CONST: class contains all the available layer types. See `const.py` for more detail. You can choose one const from `DENSE_LAYER_CONST` to initialize a specific type of layer.


### ClassifierPack

ClassifierPack contains different classifier, including `Classifier` and corresponding `const`. You can find more detail in the `ClassifierPack` folder. We have created the `Classifier` packer. Now we can implement the `kNN`, `MLP`, and `SVM` classifier by using the same interface in sklearn.

Available methods or classes are as follows:

- ClassifierPack.Classifier: `Classifier(classifier_type: str, model: BaseAE, **kwargs)` Initialize a specific type of classifier based on the classifier_type argument and additional arguments.

    - Classifier.fit: `fit(dataloader: DataLoader) -> float` Fit the classifier based on the dataloader and return the train accuracy.

    - Classifier.predict: `predict(dataloader: DataLoader) -> float` Predict the classifier based on the dataloader and return the test accuracy.

- ClassifierPack.const.CLASSIFIER_CONST: class contains all the available classifier types. See `const.py` for more detail. You can choose one const from `CLASSIFIER_CONST` to initialize a specific type of classifier.

### RetrieverPack

RetrieverPack contains different retriever, including `Retriever` and corresponding `const`. You can find more detail in the `RetrieverPack` folder. We have created the `Retriever` packer. Now we can implement the `DistanceRetriever` by using the same interface in ***faiss***.

Available methods or classes are as follows:

- RetrieverPack.Retriever: `Retriever(retriever_type: str, model: BaseAE | Reducer | None = None, **kwargs)` Initialize a specific type of retriever based on the retriever_type argument and additional arguments.

    - Retriever.evaluate: `evaluate(dataloader: DataLoader, top_K: int = 5, retrieval_N: int = 5, label_num: int = 200) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]` Evaluate the retriever based on the query_loader and retrieval_loader and return the retrieval recall, the distance matrix of query, the distance matrix of query latent, the distance matrix of retrieval, the distance matrix of retrieval latent.

- RetrieverPack.const.RETRIEVER_CONST: class contains all the available retriever types. See `const.py` for more detail. You can choose one const from `RETRIEVER_CONST` to initialize a specific type of retriever.

### DenoiserPack

DenoiserPack contains different denoiser, including `Denoiser` and corresponding `const`. You can find more detail in the `DenoiserPack` folder. We have created the `Denoiser` packer. Now we can implement the `Denoiser` by using the same interface in ***pytorch***.

Available methods or classes are as follows:

- DenoiserPack.Denoiser: `Denoiser(denoiser_type: str, model: BaseAE | Reducer | None = None, **kwargs)` Initialize a specific type of denoiser based on the denoiser_type argument and additional arguments.

    - ExpToolKit.evaluate_denoiser: `evaluate_denoiser(denoiser: Denoiser, train_loader: DataLoader, test_loader: DataLoader, is_train: bool = True, is_print: bool = True, **kwargs: Any) -> Tuple[float, float]` Evaluate the denoiser based on the train_loader and test_loader and return a tuple containing the average loss of the denoiser on the training dataset and the average loss of the denoiser on the test dataset.

- DenoiserPack.const.DENOISER_CONST: class contains all the available denoiser types. See `const.py` for more detail. You can choose one const from `DENOISER_CONST` to initialize a specific type of denoiser.

### ExpToolKit

ExpToolKit contains some tools for experiment. You can find more detail in the `ExpToolKit` folder. We have created the `create_train_setting` and `create_dataloader` function in ExpToolKit. Now we can implement the `train_and_test` function in ExpToolKit.

---

## How to use

You can create an Auto-Encoder model based on the config file. An example yaml config file is as follows:

```yaml
DATA:
  type: MNIST
MODEL:
  hidden_dims: []
  latent_dim: 16
  layer_type: LINEAR
  model_type: AE
TRAIN:
  batch_size: 256
  epochs: 10
  lr: 1e-4
  optim_type: ADAM
  random_seed: 20240
  weight_decay: 1e-4
```

All the arguments in yaml config file are as follows:

```yaml
DATA:
  type: str # MNIST, FASHION_MNIST, CIFAR10, CIFAR100

MODEL:
  hidden_dims: list
  latent_dim: int 
  layer_type: str # LINEAR, KAN, WAVELET_KAN, FOURIER_KAN, KAE
  model_type: str # AE

TRAIN:
  batch_size: int
  epochs: int
  lr: float
  optim_type: ADAM
  random_seed: int
  weight_decay: float
```

Training can be done in the following way:

```python
import ExpToolKit

config_path = "path_to_config.yaml"
config = ExpToolKit.load_config(config_path)
train_setting = ExpToolKit.create_train_setting(config)

model, train_loss_epoch, train_loss_batch, epoch_time, test_loss_epoch = (
    ExpToolKit.train_and_test(**train_setting, is_print=False)
)
```

### Evaluate Model on Various Tasks

- Evaluate the classifier on classification task:

    After we get the trained dimension reduction model, we can evaluate the performance of model on classification task by using `evaluate_classifier` function in ExpToolKit. 

    ```python
    classifier = Classifier(CLASSIFIER_CONST.KNN_CLASSIFIER, model, n_neighbors=5)
    train_accuracy, test_accuracy = etk.evaluate_classifier(classifier, train_loader, test_loader)
    ```

    Choose the classifier type from `CLASSIFIER_CONST` in `ClassifierPack.const.py` with corresponding parameters, and do not forget to pass the trained dimension reduction model.

- Evaluate the retriever on retrieval task:

    ```python
    retriever = Retriever(RETRIEVER_CONST.DISTANCE_RETRIEVER, model)
    train_recall, test_recall = etk.evaluate_retriever(retriever, train_loader, test_loader, top_K=5, retrieval_N=5, label_num=200)
    ```

    Choose the retriever type from `RETRIEVER_CONST` in `RetrieverPack.const.py` with corresponding parameters.


- Evaluate the denoiser on denoising task:

    ```python
    denoiser = Denoiser(DENOISER_CONST.DENOISER, model)
    train_loss, test_loss = etk.evaluate_denoiser(denoiser, train_loader, test_loader, is_print=True, is_train=False, epochs=2, noise_type=DENOISER_CONST.SALT_AND_PEPPER_NOISE, noise_params=(0.05, 0.95))
    ```

    Choose the denoiser type from `DENOISER_CONST` in `DenoiserPack.const.py` with corresponding parameters.

---

## How to cite

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{yu2024kae,
  title={KAE: Kolmogorov-Arnold Auto-Encoder for Representation Learning},
  author={Yu, Fangchen and Hu, Ruilizhen and Lin, Yidong and Ma, Yuqi and Huang, Zhenghao and Li, Wenye},
  journal={arXiv preprint arXiv:2501.00420},
  year={2024}
}
```

---

> *Last Update: 2025-01-08*