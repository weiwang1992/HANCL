import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1_seq = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            Conv2d(3,32,5,1,2),
            #nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            MaxPool2d(2),
            Conv2d(32,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4,64),
            Linear(64,10)
     
        )

    def forward(self, x):
        x = self.model1_seq(x)
        return x

model = Model()
input = torch.ones(64,3,32,32)
output = model(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(model, input)
writer.close()