import torch.nn as nn
import torch.nn.functional as F
import torch.legacy.nn as L



class DeepResNet101(nn.Module):

    """using bottle-neck building block """


    def __init__(self, D_out, kernel=7, padding=3):
        super(DeepResNet101, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel, padding=padding)
        self.conv2_ = nn.Conv2d(32, 32, 1, stride=2)
        self.conv2 = nn.Conv2d(128, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.conv4 = nn.Conv2d(32, 128, 5, padding=2)
        self.conv5_ = nn.Conv2d(128, 64, 1, stride=2)
        self.conv5 = nn.Conv2d(256, 64, 5, padding=2)
        self.conv6 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv7 = nn.Conv2d(64, 256, 5, padding=2)
        self.conv8_ = nn.Conv2d(256, 128, 1, stride=2)
        self.conv8 = nn.Conv2d(512, 128, 5, padding=2)
        self.conv9 = nn.Conv2d(128, 128, kernel, padding=padding)
        self.conv10 = nn.Conv2d(128, 512, 5, padding=2)
        self.conv11_ = nn.Conv2d(512, 256, 1, stride=2)
        self.conv11 = nn.Conv2d(1024, 256, 5, padding=2)
        self.conv12 = nn.Conv2d(256, 256, kernel, padding=padding)
        self.conv13 = nn.Conv2d(256, 1024, 5, padding=2)

        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(512)
        self.norm6 = nn.BatchNorm2d(1024)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        # ----> 1, 129, 129
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)              # max pooling ?   3x3 s=2

        # ----> 32, 64, 64   First Group
        x = residualLayer3(x, self.conv2_, self.conv3, self.conv4, self.norm1, self.norm3, 32, 32, 128, stride=2)
        for i in range(3-1): x = residualLayer3(x, self.conv2, self.conv3, self.conv4, self.norm1, self.norm3, 128, 32, 128)

        x = residualLayer3(x, self.conv5_, self.conv6, self.conv7, self.norm2, self.norm4, 128, 64, 256, stride=2)
        for i in range(8-1): x = residualLayer3(x, self.conv5, self.conv6, self.conv7, self.norm2, self.norm4, 256, 64, 256)

        x = residualLayer3(x, self.conv8_, self.conv9, self.conv10, self.norm3, self.norm5, 256, 128, 512, stride=2)
        for i in range(36-1): x = residualLayer3(x, self.conv8, self.conv9, self.conv10, self.norm3, self.norm5, 512, 128, 512)

        x = residualLayer3(x, self.conv11_, self.conv12, self.conv13, self.norm4, self.norm6, 512, 256, 1024, stride=2)
        for i in range(3-1): x = residualLayer3(x, self.conv11, self.conv12, self.conv13, self.norm4, self.norm6, 1024, 256, 1024)

        # ----> 1024, 4, 4   Pooling, Linear, Softmax
        x = nn.AvgPool2d(4,4)(x)
        x = x.view(-1, 1024)
        x = self.linear1(x)
        x = F.dropout(x)  # ==============================
        x = self.linear2(x)
        x = self.linear3(x)

        return x





class DeepResNet50(nn.Module):

    """using bottle-neck building block """



    def __init__(self, D_out, kernel=7, padding=3):   #=============== conv window size 5/22 9:24pm
        super(DeepResNet50, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel, padding=padding)
        self.conv2_ = nn.Conv2d(32, 32, 1, stride=2)
        self.conv2 = nn.Conv2d(128, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.conv4 = nn.Conv2d(32, 128, 5, padding=2)
        self.conv5_ = nn.Conv2d(128, 64, 1, stride=2)
        self.conv5 = nn.Conv2d(256, 64, 5, padding=2)
        self.conv6 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv7 = nn.Conv2d(64, 256, 5, padding=2)
        self.conv8_ = nn.Conv2d(256, 128, 1, stride=2)
        self.conv8 = nn.Conv2d(512, 128, 5, padding=2)
        self.conv9 = nn.Conv2d(128, 128, kernel, padding=padding)
        self.conv10 = nn.Conv2d(128, 512, 5, padding=2)
        self.conv11_ = nn.Conv2d(512, 256, 1, stride=2)
        self.conv11 = nn.Conv2d(1024, 256, 5, padding=2)
        self.conv12 = nn.Conv2d(256, 256, kernel, padding=padding)
        self.conv13 = nn.Conv2d(256, 1024, 5, padding=2)

        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(256)
        self.norm5 = nn.BatchNorm2d(512)
        self.norm6 = nn.BatchNorm2d(1024)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(3, stride=2)


    def forward(self, x):
        # ----> 1, 129, 129
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)             # ================= max pooling ?   better without here
        

        # ----> 32, 64, 64   First Group
        x = residualLayer3(x, self.conv2_, self.conv3, self.conv4, self.norm1, self.norm3, 32, 32, 128, stride=2)
        for i in range(3-1): x = residualLayer3(x, self.conv2, self.conv3, self.conv4, self.norm1, self.norm3, 128, 32, 128)

        # ----> 128, 32, 32    Second Group
        x = residualLayer3(x, self.conv5_, self.conv6, self.conv7, self.norm2, self.norm4, 128, 64, 256, stride=2)
        for i in range(4-1): x = residualLayer3(x, self.conv5, self.conv6, self.conv7, self.norm2, self.norm4, 256, 64, 256)

        # ----> 256, 16, 16    Third Group
        x = residualLayer3(x, self.conv8_, self.conv9, self.conv10, self.norm3, self.norm5, 256, 128, 512, stride=2)
        for i in range(6-1): x = residualLayer3(x, self.conv8, self.conv9, self.conv10, self.norm3, self.norm5, 512, 128, 512)

        # ----> 512, 8,8    Fourth Group
        x = residualLayer3(x, self.conv11_, self.conv12, self.conv13, self.norm4, self.norm6, 512, 256, 1024, stride=2)
        for i in range(3-1): x = residualLayer3(x, self.conv11, self.conv12, self.conv13, self.norm4, self.norm6, 1024, 256, 1024)

        # ----> 1024, 4, 4   Pooling, Linear, Softmax
        x = nn.AvgPool2d(4,4)(x)
        x = x.view(-1, 1024)
        x = self.linear1(x)
        x = F.dropout(x)    # ==============================
        x = self.linear2(x)
        x = self.linear3(x)

        return x


# stack of 3 layers providing shortcuts
def residualLayer3(input, conv2d1, conv2d2, conv2d3, norm2d1, norm2d2, inChannels, hiddenChannels, outChannels, stride=1):
    net = conv2d1(input)  # 1x1
    net = norm2d1(net)
    net = F.relu(net)
    net = F.dropout(net)      #  ========================== dropout within blocks ????  8.21 9pm

    net = conv2d2(net)    # kernel 3x3 or 5x5
    net = norm2d1(net)
    net = F.relu(net)
    net = F.dropout(net)     #  ========================== dropout   ????   8.21 9pm

    net = conv2d3(net)    # 1x1

    skip = input
    #print "input:      " + str(skip.data.size())
    if stride > 1:
        skip = L.SpatialAveragePooling(1, 1, stride, stride).forward(skip.cpu().data)
        skip = Variable(skip.cuda())
    if outChannels > inChannels:
        skip = L.Padding(1, (outChannels - inChannels), 3).forward(skip.cpu().data)
        skip = Variable(skip.cuda())
    elif outChannels < inChannels:
        skip = L.Narrow(2, 1, outChannels).forward(skip.cpu().data)
        skip = Variable(skip.cuda())

    #net = norm2d2(net)               
    #print "skip:      " + str(skip.data.size())
    #print "net:      " + str(net.data.size())
    net = norm2d2(torch.add(skip, net))    #  ==========================BN after add or before ???
    net = F.dropout(net)          #  ==========================  dropout ????   
    return  net


