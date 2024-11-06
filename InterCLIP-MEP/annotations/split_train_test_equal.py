import json
from sklearn.model_selection import train_test_split


# Load the JSON file
train_path = 'vimmsd_train_old.json'
with open(train_path, 'r', encoding='utf-8') as file:
    json_file = json.load(file)

def save_json(save_path, json_data):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

key_multi = []
key_not = []
key_text = []
key_image = []

for key, val in json_file.items():
    if val['label'] == 'multi-sarcasm':
        key_multi.append(key)
    if val['label'] == 'not-sarcasm':
        key_not.append(key)
    if val['label'] == 'image-sarcasm':
        key_image.append(key)
    if val['label'] == 'text-sarcasm':
        key_text.append(key)

key_multi_len = len(key_multi)
key_not_len = len(key_not)
key_text_len = len(key_text)
key_image_len = len(key_image)

train_key_multi, val_key_multi = key_multi[:int(key_multi_len * 0.8)], key_multi[int(key_multi_len * 0.8):]
train_key_not, val_key_not = key_not[:int(key_not_len * 0.8)], key_not[int(key_not_len * 0.8):]
# train_key_text, val_key_text = key_text[:int(key_text_len * 0.8)], key_text[int(key_text_len * 0.8):]
# train_key_image, val_key_image = key_image[:int(key_image_len * 0.8)], key_image[int(key_image_len * 0.8):]
train_key_text, val_key_text = key_text[:1300], key_text[1300:2000]
train_key_image, val_key_image = key_image[:1300], key_image[1300:2000]

train_keys = train_key_multi + train_key_not + train_key_text + train_key_image
val_keys = val_key_multi + val_key_not + val_key_text + val_key_image

# Create train and test JSON objects
train_json = {key: json_file[key] for key in list(train_keys)}
val_json = {key: json_file[key] for key in val_keys}

# Save the train and test JSON files
save_json('vimmsd_train.json', train_json)
save_json('vimmsd_val.json', val_json)
