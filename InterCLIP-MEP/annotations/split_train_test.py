import json
from sklearn.model_selection import train_test_split

# Load the JSON file
train_path = 'vimmsd_train_old.json'
with open(train_path, 'r', encoding='utf-8') as file:
    json_file = json.load(file)

# Split the data
list_keys = list(json_file.keys())
train_keys, val_keys = train_test_split(list_keys, test_size=0.1, shuffle=True)

# Create train and test JSON objects
train_json = {key: json_file[key] for key in list(train_keys)[:1000]}
val_json = {key: json_file[key] for key in val_keys}

# Define the function to save JSON data
def save_json(save_path, json_data):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

# Save the train and test JSON files

save_json('vimmsd_train.json', train_json)
save_json('vimmsd_val.json', val_json)
