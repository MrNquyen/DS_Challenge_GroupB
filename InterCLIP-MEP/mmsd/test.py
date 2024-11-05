from utils import load_config_from_yaml

if __name__=='__main__':
    path = 'best.yaml'
    config = load_config_from_yaml(path)
    print(config.keys())
