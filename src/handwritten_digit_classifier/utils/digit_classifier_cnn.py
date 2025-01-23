from torch import nn


class DigitClassifierCNN(nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        # build a convolutional neural network
        # here we are taking an image of size 28x28 with only one channel greyscale so the image dimension 1x28x28
        # here we are using kernel size of 3x3x1 and total 32 kernels this gives an output images of size 28x28 with 32 channels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) # output size: 32x28x28
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # output size: 32x14x14
        # here the input channel is 32 and output channel is 64
        # the input is the number of kernels/channels from the previous layer (the previous layer has output of 32 kernels)
        # here we are using kernel size of 3x3x32 and total 64 kernels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # output size: 64x14x14
        # here we are using the pooling layer to reduce the size of the image
        # the input channel is 64 and output channel is 64
        # before we apply this we apply pooling that reduces the image size to 7x7 so the output size is 64x7x7
        # giving the None means we are giving the value dynamically in the forward function
        self.fc1 = None # nn.Linear(64*7*7, 128) # Fully connected layer output size: 128
        self.fc2 = nn.Linear(128, 10) # Fully connected layer output size: 10 10 classes for 10 digits 0-9

    def forward(self, x):
        # define the forward network
        x = self.conv1(x) # input: 1x28x28 output: 32x28x28
        x = self.relu(x) # apply the activation function
        x = self.pool(x) # input: 32x28x28 output: 32x14x14

        x = self.conv2(x) # input: 32x14x14 output: 64x14x14
        x = self.relu(x) # apply the activation function
        x = self.pool(x) # input: 64x14x14 output: 64x7x7

        if self.fc1 is None:
            # dynamically configure the fully connected layer based on the output size of the convolutional layer
            self.fc1 = nn.Linear(x.view(x.size(0), -1), 128)
        x = x.view(x.size(0), -1) # flatten the image for the fully connected layer input: 64x7x7 output: 64*7*7 = 3136
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
