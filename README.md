# Supplementary Materials

The source code for the ICLR'2025 submission #6182, titled **"KAE: Kolmogorov-Arnold Auto-Encoder for Representation Learning"**.

## Code Structure

There are basicly 4 packer `DenseLayerPack`, `ClassifierPack`, `RetrieverPack`, `DenoiserPack` and `ExpToolKit`.

### DenseLayerPack

DenseLayerPack contains different dense layer, including `DenseLayer`, `WaveletKAN`, `TaylorKAN`, `FFTKAN`,and `ConvKAN`. You can find more detail in the `DenseLayerPack` folder. Thanks to previous work of [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/), [FourierKAN](https://github.com/GistNoesis/FourierKAN/), [Wav-KAN](https://github.com/zavareh1/Wav-KAN), [efficient-kan](https://github.com/KindXiaoming/efficient-kan) and [pykan](https://github.com/KindXiaoming/pykan), we have created the `DenseLayer` packer. Now we can implement the `WaveletKAN`, `TaylorKAN`, `FFTKAN`, and `ConvKAN` and `Linear` layer by using the same interface in pytorch. The copyright of the code belongs to the original authors which we mentioned above.

### ClassifierPack

ClassifierPack contains different classifier, including `Classifier` and corresponding `const`. You can find more detail in the `ClassifierPack` folder. We have created the `Classifier` packer with the kNN classifier.

### RetrieverPack

RetrieverPack contains different retriever, including `Retriever` and corresponding `const`. You can find more detail in the `RetrieverPack` folder. We have created the `Retriever` packer. Now we can implement the `DistanceRetriever` by using the same interface in ***faiss***.

### DenoiserPack

DenoiserPack contains different denoiser, including `Denoiser` and corresponding `const`. You can find more detail in the `DenoiserPack` folder. We have created the `Denoiser` packer. Now we can implement the `Denoiser` by using the same interface in ***pytorch***.

### ExpToolKit

ExpToolKit contains some tools for experiment. You can find more detail in the `ExpToolKit` folder. We have created the `create_train_setting` and `create_dataloader` function in ExpToolKit. Now we can implement the `train_and_test` function in ExpToolKit.

---

## How to use

You can create AutoEncoder model based on the config file. An example yaml config file is as follows:

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
train_setting = ExpToolKit.create_train_setting(config_path)
model, train_loss, test_loss = ExpToolKit.train_and_test(*train_setting, is_print=False)
```
