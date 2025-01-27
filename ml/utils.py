from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from ml.model import MACROSS

def train_macross(
    c1_kernel_size,
    c1_output_dim,
    c1_stride,
    c2_kernel_size,
    c2_output_dim,
    c2_stride,
    output_dim,
    data_path,
    epoch,
    model_path,
    signal_length,
    logger,
):
    # prepare dir for model path
    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    # seed everything
    seed_everything(seed=9876, workers=True)

    model = MACROSS(
        c1_kernel_size=c1_kernel_size,
        c1_output_dim=c1_output_dim,
        c1_stride=c1_stride,
        c2_kernel_size=c2_kernel_size,
        c2_output_dim=c2_output_dim,
        c2_stride=c2_stride,
        output_dim=output_dim,
        data_path=data_path,
        signal_length=signal_length,
    ).float()
    trainer = Trainer(
        val_check_interval=1.0,
        max_epochs=epoch,
        devices="auto",
        accelerator="auto",
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor="training_loss", mode="min", check_on_train_epoch_end=True
            )
        ],
    )
    trainer.fit(model)

    # save model
    print('model saved at:'+str(model_path.absolute()))
    trainer.save_checkpoint(str(model_path.absolute()))


def train_status_classification_macross_model(data_path, model_path):
    logger = TensorBoardLogger(
        "status_classification_macross_logs", "status_classification_macross"
    )
    train_macross(
        c1_kernel_size=4,
        c1_output_dim=50,
        c1_stride=3,
        c2_kernel_size=5,
        c2_output_dim=50,
        c2_stride=1,
        output_dim=4,
        data_path=data_path,
        epoch=20,
        model_path=model_path,
        signal_length=1500,
        logger=logger,
    )

def train_attack_classification_macross_model(data_path, model_path):
    logger = TensorBoardLogger(
        "attack_classification_macross_logs", "attack_classification_macross"
    )
    train_macross(
        c1_kernel_size=5,
        c1_output_dim=50,
        c1_stride=3,
        c2_kernel_size=4,
        c2_output_dim=50,
        c2_stride=3,
        output_dim=15,
        data_path=data_path,
        epoch=20,
        model_path=model_path,
        signal_length=1500,
        logger=logger,
    )


def train_binary_classification_macross_model(data_path, model_path):
    logger = TensorBoardLogger(
        "binary_classification_macross_logs", "binary_classification_macross"
    )
    train_macross(
        c1_kernel_size=4,
        c1_output_dim=50,
        c1_stride=3,
        c2_kernel_size=5,
        c2_output_dim=50,
        c2_stride=1,
        output_dim=2,
        data_path=data_path,
        epoch=20,
        model_path=model_path,
        signal_length=1500,
        logger=logger,
    )

def train_scenario_classification_macross_model(data_path, model_path):
    logger = TensorBoardLogger(
        "scenario_classification_macross_logs", "scenario_classification_macross"
    )
    train_macross(
        c1_kernel_size=5,
        c1_output_dim=50,
        c1_stride=3,
        c2_kernel_size=4,
        c2_output_dim=50,
        c2_stride=3,
        output_dim=3,
        data_path=data_path,
        epoch=20,
        model_path=model_path,
        signal_length=1500,
        logger=logger,
    )

def load_macross_model(model_path, gpu):
    if gpu:
        device = "cuda"
    else:
        device = "cpu"
    model = (
        MACROSS.load_from_checkpoint(
            str(Path(model_path).absolute()), map_location=torch.device(device) 
        )
        .float()
        .to(device)
    )

    model.eval()

    return model

def load_status_classification_macross_model(model_path, gpu=False):
    return load_macross_model(model_path=model_path, gpu=gpu)

def load_attack_classification_macross_model(model_path, gpu=False):
    return load_macross_model(model_path=model_path, gpu=gpu)

def load_binary_classification_macross_model(model_path, gpu=False):
    return load_macross_model(model_path=model_path, gpu=gpu)

def load_scenario_classification_macross_model(model_path, gpu=False):
    return load_macross_model(model_path=model_path, gpu=gpu)

def normalise_cm(cm):
    with np.errstate(all="ignore"):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm
