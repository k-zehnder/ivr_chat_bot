import typing as t
from loguru import logger
from pathlib import Path
import torch

# from model import NeuralNet
import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
        
class ModelLoader:

    def __init__(self, model_name, model_directory):
        self.model_name = Path(model_name)
        self.model_directory = Path(model_directory)
        self.save_path = self.model_directory / self.model_name
        self.model = self._load_model()

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[+] Model loaded in {device} complete")
        FILE = "/home/batman/Desktop/py/ivr_chat_bot/data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data['all_words']
        tags = data['tags']
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

    def predict(self):
        pass

if __name__ == "__main__":
    ml = ModelLoader("data.pt", "/home/batman/Desktop/py/ivr_chat_bot")
    print(ml.save_path)
    print(dir(ml.model))