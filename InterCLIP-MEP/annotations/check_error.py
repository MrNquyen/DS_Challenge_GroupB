import os
import json
from PIL import Image

def check(annotation='annotations', file_annotation='vimmsd_train.json', file_image='images'):
    data = []

    # Path of folder images and file annotations
    file_json = os.path.join('.', annotation, file_annotation)
    file_image = os.path.join('.', file_image)
    
    lable2id={
        "image-sarcasm": 0,
        "text-sarcasm": 1,
        "multi-sarcasm": 2,
        "not-sarcasm": 3,
    }

    with open(file_json, 'r', encoding='utf-8') as file:
        data_json = json.load(file)
    for idx, value in data_json.items():
        image_path = os.path.join(file_image, value['image'])
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            img = None
            
        if img is None:
            continue
        
        try:
            label = lable2id[value['label']]
        except:
            label = None
        # Store data as paths or preprocessed values
        data.append({
            'id': idx,
            'image': img,
            'text': value['caption'],
            'label': label,
        })

path = 'InterCLIP-MEP\\annotations\\vimmsd_test.json'
with open(path, 'r', encoding='utf-8') as file:
    data_json = json.load(file)

label = data_json['1348']['label']
print(label)
# print(data_json.keys())
