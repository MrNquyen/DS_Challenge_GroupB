import logging
from pathlib import Path
import numpy as np
import torch
from typing import cast
from pytorch_lightning.callbacks import ModelCheckpoint
from datasetv2 import SacarsmDatasetModule
from lit_modelv2 import LitSacarsmModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from utils import load_config_from_yaml
from transformers import AutoConfig
import pandas as pd

logger = logging.getLogger("mmsd")
logger.propagate = False
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

def test_model(trainer, datamodule, result_save_path=None, search_memo_size=False, memo_size_save_path=None):
    rv_file = None
    if result_save_path is not None:
        result_save_path_obj = Path(result_save_path)
        rv_file = result_save_path_obj.open("a" if result_save_path_obj.exists() else "w")

    memo_rv_file = None
    if search_memo_size and memo_size_save_path is not None:
        memo_size_save_path_obj = Path(memo_size_save_path)
        memo_rv_file = memo_size_save_path_obj.open("a" if memo_size_save_path_obj.exists() else "w")

    logger.info("Start to test the model.")
    visited_epoch = []
    max_acc = -np.inf
    max_rv = None
    max_ckpt = None
    max_memo_size = None
    for ckpt in trainer.checkpoint_callbacks:
        ckpt = cast(ModelCheckpoint, ckpt)
        p = Path(ckpt.best_model_path)

        epoch = p.name.split("-")[1].split("=")[1]
        if epoch in visited_epoch:
            continue
        visited_epoch.append(epoch)

        logger.info(
            f"Testing the model with the best checkpoint: {ckpt.best_model_path}, the best metric: {ckpt.best_model_score}"
        )

        if search_memo_size:
            memo_sizes = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
            for memo_size in memo_sizes:
                model = LitSacarsmModel.load_from_checkpoint(p, memo_size=memo_size)
                result = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt.best_model_path, verbose=False)
                if memo_rv_file is not None:
                    rv = result[0]
                    acc = round(rv["test_binary/BinaryAccuracy"] * 100, 2)
                    f1 = round(rv["test_binary/BinaryF1Score"] * 100, 2)
                    pr = round(rv["test_binary/BinaryPrecision"] * 100, 2)
                    r = round(rv["test_binary/BinaryRecall"] * 100, 2)
                    memo_rv_file.write(f"{memo_size}    {acc}    {f1}    {pr}    {r}\n")
                if result[0]["test_binary/BinaryAccuracy"] > max_acc:
                    max_acc = result[0]["test_binary/BinaryAccuracy"]
                    max_rv = result[0]
                    max_ckpt = ckpt.best_model_path
                    max_memo_size = memo_size
        else:
            result = trainer.test(model, datamodule=datamodule, ckpt_path=ckpt.best_model_path, verbose=False)
            if result[0]["test_binary/BinaryAccuracy"] > max_acc:
                max_acc = result[0]["test_binary/BinaryAccuracy"]
                max_rv = result[0]
                max_ckpt = ckpt.best_model_path

    if rv_file is not None:
        acc = round(max_rv["test_binary/BinaryAccuracy"] * 100, 2)
        f1 = round(max_rv["test_binary/BinaryF1Score"] * 100, 2)
        p = round(max_rv["test_binary/BinaryPrecision"] * 100, 2)
        r = round(max_rv["test_binary/BinaryRecall"] * 100, 2)
        rv_file.write(f"{max_ckpt}    {acc}    {f1}    {p}    {r}\n")
        rv_file.close()

def cli_main() -> None:
    device = "gpu" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("medium")
    config_yaml = load_config_from_yaml(file_path='mmsd/best.yaml')
    # visobert_config = AutoConfig()
    
    # Initialize dataset and model
    vision_ckpt_name = 'openai/clip-vit-base-patch32'
    text_ckpt_name = 'uitnlp/visobert'
    # text_ckpt_name = 'openai/clip-vit-base-patch32'
    datamodule = SacarsmDatasetModule(
        vision_ckpt_name=vision_ckpt_name,
        text_ckpt_name=text_ckpt_name,
    )
    config_yaml_model = config_yaml['model']
    clip_ckpt_name = config_yaml_model['clip_ckpt_name']
    vision_embed_dim = config_yaml_model['vision_embed_dim']
    text_embed_dim = config_yaml_model['text_embed_dim']
    vision_num_layers = config_yaml_model['vision_num_layers']
    text_num_layers = config_yaml_model['text_num_layers']
    
    # VisoBERT config
    

    print('Load LitSacarsmModel')
    print(f'Load clip_ckpt_name: {clip_ckpt_name}')
    model = LitSacarsmModel(
        vision_ckpt_name=clip_ckpt_name,
        text_ckpt_name='uitnlp/visoBERT',
        vision_embed_dim=vision_embed_dim,
        text_embed_dim=text_embed_dim,
        vision_num_layers=vision_num_layers,
        text_num_layers=text_num_layers,
    )

    # Configure the checkpoint callback to save the best model
    checkpoint_callback_train_f1 = ModelCheckpoint(
        monitor="train/MulticlassF1Score",   # or any other validation metric you are tracking
        save_top_k=1,
        mode="min",
        filename="{epoch}-{train_MulticlassF1Score:.2f}"
    )
    
    checkpoint_callback_val_f1 = ModelCheckpoint(
        monitor="val/MulticlassF1Score",   # or any other validation metric you are tracking
        save_top_k=1,
        mode="min",
        filename="{epoch}-{val_MulticlassF1Score:.2f}"
    )

    # Initialize trainer with desired configurations
    trainer = Trainer(
        # devices=1 if device == "gpu" else 1,   # number of GPUs or CPU
        # strategy="ddp" if device == "gpu" else None,
        devices=max(1, torch.cuda.device_count()),   # number of GPUs or CPU
        strategy="ddp",
        accelerator=device, 
        callbacks=[PrintCallback(), checkpoint_callback_train_f1, checkpoint_callback_val_f1],
        max_epochs=10,    # Set the maximum epochs for training
        log_every_n_steps=50
    )

    # Start training with the datamodule
    trainer.fit(model, datamodule=datamodule)
    
    test_dataloader = datamodule.test_dataloader()
    val_dataloader = datamodule.val_dataloader()
    
    # Test Model
    results = trainer.test(
        model,
        dataloaders=test_dataloader,
        verbose=True,
    )

    print(results)
    # # Call the test function with desired arguments
    # print(f'Test Model Here')
    # test_model(
    #     trainer, 
    #     datamodule, 
    #     result_save_path="results.txt",          # Set the path to save results
    #     search_memo_size=True, 
    #     memo_size_save_path="memo_sizes.txt"     # Set the path to save memo size results
    # )

if __name__ == "__main__":
    cli_main()
