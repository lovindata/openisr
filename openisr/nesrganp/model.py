import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        # Conv2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, \
            dilation=1, groups=1, bias=True)
            # stride=1
            # padding=1
            # dilation=1 (no dilatation) Spacing between kernel elements
            # learnable bias to the output

        # ResidualDenseBlock
        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, bias=False) # 64 -> 32
        self.conv2_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, \
            dilation=1, bias=True, groups=1)
        self.act2_2 = nn.LeakyReLU(neg_slope=0.2, inplace=True) # max(0,x) + negative_slope∗min(0,x)

        # WIP
