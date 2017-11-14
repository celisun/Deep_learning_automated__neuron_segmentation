import torch.nn as nn
import torch.nn.functional as F
import torch.legacy.nn as L


def residualLayer2(conv2d1, norm2d, input, nChannels, nOutChannels=False, stride=1, conv2d2=False):
    """ Deep Residual Network
        https://github.com/gcr/torch-residual-networks   
        
        giving stack of 2 layers as a block providing shortcuts."""
        
        
    if not nOutChannels:
        nOutChannels = nChannels
    if not conv2d2:
        conv2d2 = conv2d1

    # part 1: conv
    net = conv2d1(input)
    net = norm2d(net)  # learnable parameters
    net = F.relu(net)
    net = conv2d2(net)


    # part 2: identity / skip connection
    skip = input
    if stride > 1:   # optional downsampling
        skip = L.SpatialAveragePooling(1, 1, stride, stride).forward(skip.cpu().data)
        skip = Variable(skip.cuda())
    if nOutChannels > nChannels:    # optional padding
        skip = L.Padding(1, (nOutChannels - nChannels), 3).forward(skip.cpu().data)
        skip = Variable(skip.cuda())
    elif nOutChannels < nChannels:   # optional narrow
        skip = L.Narrow(2, 1, nOutChannels).forward(skip.cpu().data)
        skip = Variable(skip.cuda())


    # H(x) + x
    net = norm2d(net)
    #print "skip:      " + str(skip.data.size())
    #print "net:      " + str(net.data.size())
    net = torch.add(skip, net)
    # net = F.relu(net)           # relu here ? see: http://www.gitxiv.com/comments/7rffyqcPLirEEsmpX
    #net = norm2d(net)             #  ==========================BN after add or before ???
 
    return net







class DeepResNet18(nn.Module):
    def __init__(self, D_out, kernel=3, padding=1):
        super(DeepResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel, padding=padding)
        self.conv2 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.conv3 = nn.Conv2d(32, 64, kernel, stride =2, padding=padding)
        self.conv4 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv5 = nn.Conv2d(64, 128, kernel, stride =2, padding=padding)
        self.conv6 = nn.Conv2d(128, 128, kernel, padding=padding)
        self.conv7 = nn.Conv2d(128, 256, kernel, stride =2, padding=padding)
        self.conv8 = nn.Conv2d(256, 256, kernel, padding=padding)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        # ----> 1, 33, 33
        x = F.relu(self.norm1(self.conv1(x)))

        # ----> 32, 33, 33   First Group 2X
        for i in range(2): x = residualLayer2(self.conv2, self.norm1, x, 32)

        # ----> 64, 17, 17    Second Group 2X
        x = residualLayer2(self.conv3, self.norm2, x, 32, 64, stride=2, conv2d2=self.conv4)
        for i in range(2-1): x = residualLayer2(self.conv4, self.norm2, x, 64)

        # ----> 128, 9, 9    Third Group 2X
        x = residualLayer2(self.conv5, self.norm3, x, 64, 128, stride=2, conv2d2=self.conv6)
        for i in range(2-1): x = residualLayer2(self.conv6, self.norm3, x, 128)

        # ----> 256, 5, 5    Fourth Group 2X
        x = residualLayer2(self.conv7, self.norm4, x, 128, 256, stride=2, conv2d2=self.conv8)
        for i in range(2-1): x = residualLayer2(self.conv8, self.norm4, x, 256)

        # ----> 256, 5, 5   Pooling, Linear, Softmax
        x = nn.AvgPool2d(5,5)(x)
        x = x.view(-1, 256)
        x = self.linear(x)


        return x


class DeepResNet34(nn.Module):
    def __init__(self, D_out, kernel=5, padding=2):
        super(DeepResNet34, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel, padding=padding)
        self.conv2 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.conv3 = nn.Conv2d(32, 64, kernel, stride =2, padding=padding)
        self.conv4 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv5 = nn.Conv2d(64, 128, kernel, stride =2, padding=padding)
        self.conv6 = nn.Conv2d(128, 128, kernel, padding=padding)
        self.conv7 = nn.Conv2d(128, 256, kernel, stride =2, padding=padding)
        self.conv8 = nn.Conv2d(256, 256, kernel, padding=padding)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.linear = nn.Linear(256, 2)
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
	    # ------>  65 * 65
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)      # ================= max pooling  ??  
        # ------>  32 * 32
        for i in range(3): x = residualLayer2(self.conv2, self.norm1, x, 32)
        # ------>  32 * 32
        x = residualLayer2(self.conv3, self.norm2, x, 32, 64, stride=2, conv2d2=self.conv4)
        for i in range(4-1): x = residualLayer2(self.conv4, self.norm2, x, 64)

        x = residualLayer2(self.conv5, self.norm3, x, 64, 128, stride=2, conv2d2=self.conv6)
        for i in range(6-1): x = residualLayer2(self.conv6, self.norm3, x, 128)

        x = residualLayer2(self.conv7, self.norm4, x, 128, 256, stride=2, conv2d2=self.conv8)
        for i in range(3-1): x = residualLayer2(self.conv8, self.norm4, x, 256)

        x = nn.AvgPool2d(8,8)(x)
        x = x.view(-1, 256)
        x = self.linear(x)


        return x
