import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=[1,1]),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=[1,1]),
            torch.nn.ReLU(),
        )
        self.first_linear = torch.nn.Linear(in_features=131072, out_features=1024)
        self.second_linear = torch.nn.Linear(in_features=1024, out_features=512)
        self.out_layer = torch.nn.Linear(in_features=512, out_features=4)
        
    def forward(self, x):
        b, c, h, w = x.shape

        x = self.feature_extractor(x)
        x = x.view((b, -1))
        x = self.first_linear(x)
        x = torch.relu(x)
        x = self.second_linear(x)
        x = torch.relu(x)
        x = self.out_layer(x)
        return x
