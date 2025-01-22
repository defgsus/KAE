from .training import Experiment


DEFAULT_EPOCHS = 10


EXPERIMENTS = [
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"hidden_dims": []},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"hidden_dims": [], "activation": "relu"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"hidden_dims": [], "activation": ["none", "relu"]},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"hidden_dims": [], "activation": "relu6"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 32},
        model_config={"hidden_dims": [], "activation": "relu6"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"hidden_dims": [], "activation": "relu6"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 128},
        model_config={"hidden_dims": [], "activation": "relu6"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"hidden_dims": [64]},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"hidden_dims": [64], "activation": "relu6"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"hidden_dims": [128], "activation": "relu6"},
    ),
    Experiment(
        model="MLP",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"hidden_dims": [256], "activation": "relu6"},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 3},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 4},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 5},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 6},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 3, "activation": ["relu6", "sigmoid"]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 3, "activation": ["none", "sigmoid"]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"order": 3, "activation": ["none", "sigmoid"]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS, "batch_size": 64},
        model_config={"order": 3},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS, "batch_size": 512},
        model_config={"order": 3},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"order": 3, "activation": ["relu6", "relu6"]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS, "optim_type": "ADAMW", "lr": 0.0003, "batch_size": 64},
        model_config={"order": 3},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 2, "hidden_dims": [64]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 2, "hidden_dims": [128]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 2, "hidden_dims": [256]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 3, "hidden_dims": [64]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 3, "hidden_dims": [128]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 3, "hidden_dims": [256]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 4, "hidden_dims": [64]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 4, "hidden_dims": [128]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 4, "hidden_dims": [256]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 5, "hidden_dims": [64]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 5, "hidden_dims": [128]},
    ),
    Experiment(
        model="KAE",
        train_config={"epochs": DEFAULT_EPOCHS},
        model_config={"order": 5, "hidden_dims": [256]},
    ),
]
