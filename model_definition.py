import torch.nn as nn
import torch
import torch.nn.functional as F

# CNN modelini tanımlama
class CNNModel(nn.Module):
    def __init__(self, input_length, num_classes):  # Düzeltildi
        super(CNNModel, self).__init__()  # Düzeltildi
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.pool = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear((input_length // 2) * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
