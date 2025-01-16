from torch import nn
import torch.nn.functional as F


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # flatten the image
        # -1 refers to all the images in the batch
        x = x.reshape(-1, 28*28) # flatten the image
        # x = x.view(-1, 28*28) # flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
