from torch import nn


class DigitClassifierCNNSequential(nn.Module):
    def __init__(self):
        super(DigitClassifierCNNSequential, self).__init__()
        self.conv_layers = nn.Sequential(
            # build a convolutional neural network
            # here we are taking an image of size 28x28 with only one channel greyscale so the image dimension 1x28x28
            # here we are using kernel size of 3x3x1 and total 32 kernels this gives an output images of size 28x28 with 32 channels
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # output size: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output size: 32x14x14
            # here the input channel is 32 and output channel is 64
            # the input is the number of kernels/channels from the previous layer (the previous layer has output of 32 kernels)
            # here we are using kernel size of 3x3x32 and total 64 kernels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # output size: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output size: 64x7x7
        )

        # here we are using the pooling layer to reduce the size of the image
        # the input channel is 64 and output channel is 64
        # this will perform average pooling on an input, reducing its size while preserving spatial information
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = self.fc_layers(x)
        return x
