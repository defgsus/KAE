## Additional experiments

Results are published here: https://defgsus.github.io/nn-experiments/html/logs/review-kan-autoencoder.html

This is just an add-on to the existing repo. I tried to minimize the changes to the original code.
Just had to add configurable activation functions to the `DenseLayerPack/KAE.py` 

Configuration of experiments is in `config.py`

To train all models with default settings (l2 reconstruction on MNIST, 16-dim latent vector):

```shell
# in project root
python defgsus train
```

This will store all train & test results along with the trained models in `defgsus/experiments`.

To run the additional tests
```shell
python defgsus test
```

And printing the tables and text output and rendering reproduction images:

```shell
python defgsus markdown -t
python defgsus reproductions
```
