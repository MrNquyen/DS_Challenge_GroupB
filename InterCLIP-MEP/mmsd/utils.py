import yaml

# Load yaml file
class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)  # Load YAML file into a dictionary
    # config = Config(**yaml_data)  # Pass the dictionary as keyword arguments
    return yaml_data



