import glob
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


class LinearNN(nn.Module):

    def __init__(self, num_classes=3):
        super(LinearNN, self).__init__()
        self.lin1 = nn.Linear(9, 144)
        self.lin2 = nn.Linear(144, 1000)
        self.lin3 = nn.Linear(1000, 2048)
        self.lin4 = nn.Linear(2048, 500)
        self.lin5 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        return x  # Not using softmax -> faster learning

class CustomDataset(Dataset):



    def __init__(self, data, is_train=True):
        self.master_data = []  # array of boards with their labels like this -> Boards[([Board Matrix Flattened], [Label]), ([...])]
        self.game_data = data
        # Unwrap data into master data array
        self.unwrap_games()
        # for item in self.master_data:
        #     print(item)
        print("Total Dataset: " + str(len(self.master_data)))
        self.is_train = is_train
        if self.is_train:
            self.master_data = self.master_data[:round(len(self.master_data)*3/4)]
        else:
            self.master_data = self.master_data[round(len(self.master_data)*3/4):]
        print("Cut Dataset: " + str(len(self.master_data)))

    def __len__(self):
        return len(self.master_data)

    def __getitem__(self, idx):
        data, label = self.master_data[idx]
        data_tensor = torch.from_numpy(data).float()
        label_tensor = torch.from_numpy(label).long()
        return data_tensor, label_tensor

    def unwrap_games(self):
        for game in self.game_data:
            # if game[1][0] == 1:  # If it's a draw, don't include it as over time it should even out. -> Has lowest loss if draw and it predicts 1
            #     continue
            for board in game[0]:
                self.master_data.append((np.array(board).flatten(), np.array(game[1][0])))

    def balance_master_data(self):  # Balance zero games with 1 games with 2 games
        zero_list = []
        one_list = []
        two_list = []
        for item in self.master_data:
            board, label = item
            if label == 0:
                zero_list.append(item)
            elif label == 1:
                one_list.append(item)
            elif label == 2:
                two_list.append(item)
        self.master_data.clear()
        self.master_data.append(zero_list)
        self.master_data.append(one_list[:len(zero_list)])  # zero list is always shorter than 1 and -1
        self.master_data.append(two_list[:len(zero_list)])  # zero list is always shorter than 1 and -1

class TrainData:

    save_path = "model_with_ties.pth"

    def __init__(self, data, num_epochs=2, batch_size=512, learning_rate=0.000005):
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.custom_dataset_train = CustomDataset(self.data, True)
        self.custom_dataset_test = CustomDataset(self.data, False)
        self.model = LinearNN()
        self.model = self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.CrossEntropyLoss()

        self.train_loader = DataLoader(dataset=self.custom_dataset_train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.custom_dataset_test, batch_size=self.batch_size, shuffle=True)


        # Loop
        for epoch in range(self.num_epochs):
            self.train(epoch)
            self.test()

        # Save Model
        torch.save(self.model, self.save_path)

    def train(self, epoch):
        for batch_index, (data, label) in enumerate(self.train_loader):
            data = data.to(device)
            label = label.to(device)

            result = self.model(data)
            loss = self.loss_function(result, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 0:
                print(f'Epoch = {epoch}/{self.num_epochs} Batch Index = {batch_index}/{round(len(self.train_loader))} Loss = {loss.item()}')

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_index, (data, label) in enumerate(self.test_loader):
                data = data.to(device)
                labels = label.to(device)
                # calculate outputs by running images through the network
                results = self.model(data)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(results, 1)
                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        print('Accuracy of the network: %d %%' % (
                100 * correct / total))

