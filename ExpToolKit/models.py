from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from DenseLayerPack import DenseLayer, DENSE_LAYER_CONST


class BaseAE(nn.Module):

    def __init__(self):
        """
        Initialize the BaseAE model.

        This is the base class for all autoencoder model. All autoencoder model should subclass this class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(BaseAE, self).__init__()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The preprocessed data.
        """
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def training_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Calculate the training loss.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of the training data.

        Returns
        -------
        torch.Tensor
            The training loss.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def testing_loss(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Calculate the testing loss.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of the testing data.

        Returns
        -------
        torch.Tensor
            The testing loss.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be encoded.

        Returns
        -------
        torch.Tensor
            The encoded data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def save_model(self, path: str) -> None:
        """
        Save the model to a given path.

        Parameters
        ----------
        path : str
            The path where the model is saved.

        Returns
        -------
        None
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load the model from a given path.

        Parameters
        ----------
        path : str
            The path where the model is saved.

        Returns
        -------
        None
        """
        self.load_state_dict(torch.load(path))

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """
        Reduce the dimension of the given data.

        Parameters
        ----------
        x : np.ndarray
            The input data to be reduced.

        Returns
        -------
        np.ndarray
            The reduced data.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class StandardAE(BaseAE):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        layer_type: str = DENSE_LAYER_CONST.LINEAR_LAYER,
        activation: Union[None, str, Tuple[str, str]] = None,
        **kwargs: dict
    ):
        """
        Initialize a standard AutoEncoder model.

        Parameters
        ----------
        input_dim : int
            Input dimensionality.
        hidden_dims : list
            List of hidden layer dimensionalities.
        latent_dim : int
            Latent space dimensionality.
        layer_type : str, optional
            Type of the layer to initialize. Defaults to
            DENSE_LAYER_CONST.LINEAR_LAYER.
        **kwargs : dict
            Additional arguments specific to the layer type.
        """
        super(StandardAE, self).__init__()

        if activation is None:
            act, final_act = "relu", "sigmoid"
        elif isinstance(activation, str):
            act, final_act = activation, activation
        else:
            act, final_act = activation

        encoder_layers = []
        if len(hidden_dims) == 0:
            encoder_layers.append(
                DenseLayer(input_dim, latent_dim, layer_type, **kwargs)
            )
            if (act_module := get_activation(act)) is not None:
                encoder_layers.append(act_module)
        else:
            temp_input_dim = input_dim
            for h_dim in hidden_dims:
                encoder_layers.append(
                    DenseLayer(temp_input_dim, h_dim, layer_type, **kwargs)
                )
                if (act_module := get_activation(act)) is not None:
                    encoder_layers.append(act_module)
                temp_input_dim = h_dim
            encoder_layers.append(
                DenseLayer(temp_input_dim, latent_dim, layer_type, **kwargs)
            )
            if (act_module := get_activation(act)) is not None:
                encoder_layers.append(act_module)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        if len(hidden_dims) == 0:
            decoder_layers.append(
                DenseLayer(latent_dim, input_dim, layer_type, **kwargs)
            )
            if (act_module := get_activation(final_act)) is not None:
                decoder_layers.append(act_module)
        else:
            temp_input_dim = latent_dim
            for h_dim in reversed(hidden_dims):
                decoder_layers.append(
                    DenseLayer(temp_input_dim, h_dim, layer_type, **kwargs)
                )
                if (act_module := get_activation(act)) is not None:
                    decoder_layers.append(act_module)
                temp_input_dim = h_dim
            decoder_layers.append(
                DenseLayer(temp_input_dim, input_dim, layer_type, **kwargs)
            )
            if (act_module := get_activation(final_act)) is not None:
                decoder_layers.append(act_module)
        self.decoder = nn.Sequential(*decoder_layers)

        self.latent_dim = latent_dim

    def forward(self, x) -> torch.Tensor:
        x = self.preprocess(x)
        z = self.encoder(x)
        return self.decoder(z)

    def training_loss(self, batch) -> torch.Tensor:
        x = self.preprocess(batch)
        recon_x = self.forward(x)
        loss = F.mse_loss(recon_x, x)
        return loss

    def testing_loss(self, batch) -> torch.Tensor:
        x = self.preprocess(batch)
        with torch.no_grad():
            recon_x = self.forward(x)
            loss = F.mse_loss(recon_x, x)
        return loss

    def encode(self, x) -> torch.Tensor:
        x = self.preprocess(x)
        z = self.encoder(x)
        return z

    def reduce(self, x: np.ndarray) -> np.ndarray:
        z = self.encode(torch.from_numpy(x).float())
        z = z.detach().numpy()
        return z


def get_activation(activation: str, inplace: bool = True) -> Optional[nn.Module]:
    activation = activation.lower()

    if activation == "none":
        return

    for name in dir(nn):
        if name.lower() == activation:
            klass = getattr(nn, name)
            if issubclass(klass, nn.Module):
                try:
                    return klass(inplace=inplace)
                except TypeError:
                    return klass()

    raise ValueError(f"Unknown activation '{activation}'")

