import json
json_path = 'annotations\\vimmsd_train.json'
with open(json_path, 'r', encoding='utf-8') as file:
    json_file = json.load(file)
print(len(json_file))