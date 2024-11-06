import os
import json
from functools import partial
from typing import TypedDict, cast
from PIL import Image

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
# from datasets import load_dataset
# from torch.utils.data import Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer


class SacarsmModelInput(TypedDict, total=False):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor
    id: list[str]

def preprocess(example, image_processor, tokenizer):
    image = example['image']
    text = example['text']

    image_inputs = image_processor(images=image)
    # text_inputs = tokenizer(
    #     text=text,
    #     truncation=True,
    #     padding="max_length",
    # )
    
    text_inputs = tokenizer(
        text,
        padding='max_length',  # Pad to max_length
        truncation=True,        # Truncate if longer than max_length
        max_length=77,  # Set your desired max length
        return_tensors='pt'    # Return as PyTorch tensors
    )
    
    # text_inputs = tokenizer(
    #     text,
    #     padding='max_length',  # Pad to max_length
    #     truncation=True,        # Truncate if longer than max_length
    #     # max_length=512,  # Set your desired max length
    #     return_tensors='pt'    # Return as PyTorch tensors
    # )
    
    print(len(text_inputs["input_ids"]))

    return {
        "pixel_values": image_inputs["pixel_values"],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "label": example["label"],
        "id": example["id"],
    }

class SarcasmDataLoader:
    def __init__(self, annotation='annotations', file_annotation='vimmsd_train.json', file_image='images'):
        self.data = []

        # Path of folder images and file annotations
        self.file_json = os.path.join('.', annotation, file_annotation)
        self.file_image = os.path.join('.', file_image)
        
        lable2id={
            "image-sarcasm": 0,
            "text-sarcasm": 1,
            "multi-sarcasm": 2,
            "not-sarcasm": 3,
        }

        with open(self.file_json, 'r', encoding='utf-8') as file:
            data_json = json.load(file)
        for idx, value in data_json.items():
            image_path = os.path.join(self.file_image, value['image'])
            try:
                img = Image.open(image_path).convert("RGB")
            except:
                img = None
                
            if img is None:
                continue
            
            # Store data as paths or preprocessed values
            self.data.append({
                'id': idx,
                'image': img,
                'text': value['caption'],
                'label': lable2id[value['label']]
            })

    def to_hf_dataset(self):
        # Convert the list of dictionaries to Hugging Face Dataset format
        hf_data = {
            'id': [item['id'] for item in self.data],
            'image': [item['image'] for item in self.data],
            'text': [item['text'] for item in self.data],
            'label': [item['label'] for item in self.data]
        }
        return Dataset.from_dict(hf_data)
        
        

class SacarsmDataset(Dataset):
    def __init__(self, annotation = 'annotations', file_annotation = 'vimmsd_train.json', file_image = 'images'):
        super().__init__()
        
        dataload = SarcasmDataLoader()
        sacrasm_dataload = dataload.to_hf_dataset()
        
        self.__data = {}
        for i_th, id in enumerate(load_data['id']):
            self.__data[i_th] = {
                'id': id,
                'image': sacrasm_dataload['image']['i_th'],
                'text': sacrasm_dataload['text']['i_th'],
                'label': sacrasm_dataload['label']['i_th']
            }
        
        
        print(self.__data)
        
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        return self.__data[index]
        
        
# class SacarsmDataset(Dataset):
#     def __init__(self, annotation = 'annotations', file_annotation = 'vimmsd_train.json', file_image = 'images'):
#         super(SacarsmDataset, self).__init__()
#         self.__data = {}
        
#         # Path of folder images and file annotations
#         self.file_json = os.path.join('.', annotation, file_annotation)
#         self.file_image = os.path.join('.', file_image)
        
#         with open(self.file_json, 'r', encoding = 'utf-8') as file:
#             data_json = json.load(file)
#         for i_th, (idx, value) in enumerate(data_json.items()):
#             image_path = os.path.join(self.file_image, value['image'])
#             try:
#                 img = Image.open(image_path).convert("RGB")
#             except:
#                 img = None
                
#             if img == None:
#                 continue
            
#             self.__data[i_th] = {
#                 'id': idx,
#                 'image': img,
#                 'text': value['caption'],
#                 'label': value['label']
#             }
            
#     def __len__(self):
#         return len(self.__data)
    
#     def __getitem__(self, index):
#         return self.__data[index]
    

class SacarsmDatasetModule(pl.LightningDataModule):
    def __init__(
        self,
        vision_ckpt_name: str,
        text_ckpt_name: str,
        dataset_version: str = "mmsd-v2",
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 19,
    ) -> None:
        super().__init__()
        self.vision_ckpt_name = vision_ckpt_name
        # self.text_ckpt_name = 'uitnlp/visobert'
        self.text_ckpt_name = text_ckpt_name

        self.dataset_version = dataset_version
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        image_processor = AutoImageProcessor.from_pretrained(self.vision_ckpt_name)
        tokenizer = AutoTokenizer.from_pretrained(self.text_ckpt_name)

        sacarsm_dataload_train = SarcasmDataLoader()
        sacarsm_dataload_val = SarcasmDataLoader(file_annotation='vimmsd_val.json')
        sacarsm_dataload_test = SarcasmDataLoader(file_annotation='vimmsd_test.json')
        
        self.dataset = {} 
        self.dataset['train'] = sacarsm_dataload_train.to_hf_dataset()
        self.dataset['validation'] = sacarsm_dataload_val.to_hf_dataset()
        self.dataset['test'] = sacarsm_dataload_test.to_hf_dataset()
        # self.dataset = SacarsmDataset()
        
        # print(self.dataset[0])
        for set_name in ['train', 'validation', 'test']:
            self.dataset[set_name].set_transform(
                partial(
                    preprocess,
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                )
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["train"],  # type: ignore
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["validation"],  # type: ignore
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset["test"],  # type: ignore
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, batch) -> SacarsmModelInput:
        return {
            "pixel_values": torch.stack(
                [torch.tensor(x["pixel_values"]) for x in batch]
            ),
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack(
                [torch.tensor(x["attention_mask"]) for x in batch]
            ),
            "label": torch.stack([torch.tensor(x["label"]) for x in batch]),
            "id": [x["id"] for x in batch],
        }