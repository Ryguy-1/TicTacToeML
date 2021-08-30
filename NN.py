import glob
import javaobj # For Reading Serialized Java Files
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image
import keyboard
import random


class CustomDataset(Dataset):

    master_data = [] # array of boards with their labels like this -> Boards[([Board Matrix Flattened], [Label]), ([...])]

    def __init__(self, data):
        self.game_data = data
        # Unwrap data into master data array
        self.unwrap_games()
        # for item in self.master_data:
        #     print(item)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # print(tensor.shape)
        return tensor, label_tensor

    def unwrap_games(self):
        for game in self.game_data:
            for board in game[0]:
                self.master_data.append((np.array(board).flatten(), np.array(game[1][0])))

class TrainData:

    def __init__(self, data, num_epochs=10, batch_size=32, learning_rate=0.01):
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.custom_dataset = CustomDataset(self.data)


    def train(self):
        pass









