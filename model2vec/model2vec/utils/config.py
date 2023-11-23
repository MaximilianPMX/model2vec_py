import json

class Config:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
            print(f"Config file not found at {self.config_path}. Using default configuration.")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save()

    def save(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

if __name__ == '__main__':
    config = Config()
    print(f'Current learning rate: {config.get("learning_rate", 0.001)}')
    config.set("learning_rate", 0.002)
    print(f'Updated learning rate: {config.get("learning_rate", 0.001)}')