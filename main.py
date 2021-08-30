from GameGenerator import Generator
from NN import TrainData
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

device = torch.device("cpu")
if torch.cuda.is_available():
    print(f'Switched to Cuda {torch.cuda_version}')
    device = torch.device("cuda")


def main():
    # game_generator = Generator(500000)  # 500000 works very well
    # print(f'Game Generator Length: {len(game_generator)}')
    # game_generator.printGame(game_generator.getGameAtIndex(0))
    # train_data = TrainData(game_generator.completed_games)

    testBoard([0, 0, 0,
               0, 0, 0,
               0, 0, 0])


def testBoard(board):  # Flattened Board
    model = torch.load(TrainData.save_path)
    model = model.to(device)
    board_tensor = torch.tensor(board).float()
    board_tensor = board_tensor.to(device)
    result = model(board_tensor)
    print(F.softmax(result))


if __name__ == '__main__':
    main()
