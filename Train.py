import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.legacy.nn as L
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
import numpy as np
from tempfile import TemporaryFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import rotate
import time
import os
import random

from Annotations import *
from Volume import *
from CremiFile import *
from voi import voi
from rand import adapted_rand
plt.style.use('ggplot')



print('')
print('DATASET LOADING ...')
print('')
emdset = CremiFile('sample_B_20160501.hdf', 'r')

#Check the content of the datafile
print "Has raw: " + str(emdset.has_raw())
print "Has neuron ids: " + str(emdset.has_neuron_ids())
print "Has clefts: " + str(emdset.has_clefts())
print "Has annotations: " + str(emdset.has_annotations())


#Read volume and annotation
raw = emdset.read_raw()
neuron_ids = emdset.read_neuron_ids()
clefts = emdset.read_clefts()
annotations = emdset.read_annotations()

print("")
print "Read raw: " + str(raw) + \
    ", resolution " + str(raw.resolution) + \
    ", offset " + str(raw.offset) + \
    ", data size " + str(raw.data.shape) + \
    ("" if raw.comment == None else ", comment \"" + raw.comment + "\"")

print "Read neuron_ids: " + str(neuron_ids) + \
    ", resolution " + str(neuron_ids.resolution) + \
    ", offset " + str(neuron_ids.offset) + \
    ", data size " + str(neuron_ids.data.shape) + \
    ("" if neuron_ids.comment == None else ", comment \"" + neuron_ids.comment + "\"")

print "Read clefts: " + str(clefts) + \
    ", resolution " + str(clefts.resolution) + \
    ", offset " + str(clefts.offset) + \
    ", data size " + str(clefts.data.shape) + \
    ("" if clefts.comment == None else ", comment \"" + clefts.comment + "\"")












def mask_filtered(raw, neuron_ids):
    """
    Image boudnary dilation
    Compute mask on each depth for un-selectable dilated(6X) boundary pixels (labeled as value 200.),
    the selectable background (0.) and actual boundary (100.) pixels

    return(numpy array): mask of shape 125,1250,1250
    """
    print ''
    print ''
    print 'building mask-5X from raw dataset...'
    since = time.time()

    d, h, w = raw.data.shape
    mask = np.empty([d, h, w]).astype('float32')
    for i in range(d):
        for j in range(h):
            for k in range(w):
                pixel = neuron_ids.data[i, j, k]
                if check_boundary(pixel, i, j, k, neuron_ids):
                    mask[i, j, k] = 100
                else:
                    mask[i, j, k] = 0
        if (i + 1) % 1 == 0:
            print str(0.8 * (i + 1)) + '% done'

    mask_dilated = ndimage.binary_dilation(mask, iterations=7).astype(mask.dtype)
    mask_filtered = 200 * mask_dilated - mask

    filter_time = time.time()
    time_elapsed = filter_time - since
    print('Mask complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    print 'save to maskfile7X.npy'
    np.save('maskfile7X.npy', mask_filtered)
    print 'saved'




def check_boundary(pixel, x, y, z, neuron_ids):
    """
    Check if a pixel at position (x,y,z) is labeled
    as boundary/non-boundary in neuron_ids.

    return(boolean): boundary
    """
    max_z = neuron_ids.data.shape[2] - 1
    max_y = neuron_ids.data.shape[1] - 1
    a = neuron_ids.data[x, y, z - 1] if z > 0 else pixel
    b = neuron_ids.data[x, y, z + 1] if z < max_z else pixel
    c = neuron_ids.data[x, y - 1, z] if y > 0 else pixel
    d = neuron_ids.data[x, y + 1, z] if y < max_y else pixel
    e = neuron_ids.data[x, y - 1, z - 1] if (y > 0 and z > 0) else pixel
    f = neuron_ids.data[x, y - 1, z + 1] if (y > 0 and z < max_z) else pixel
    g = neuron_ids.data[x, y + 1, z - 1] if (y < max_y and z > 0) else pixel
    h = neuron_ids.data[x, y + 1, z + 1] if (y < max_y and z < max_z) else pixel

    neighbors = [a, b, c, d, e, f, g, h]
    boundary = False
    for neighbor in neighbors:
        if pixel != neighbor:
            boundary = True

    return boundary


# Seed a random number generator
#seed = 24102016
#rng = np.random.RandomState(seed)
def random_rotation(inputs):
    """Randomly rotates a subset of images in a batch.
       reference: https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2016-7/master/notebooks/05_Non-linearities_and_regularisation.ipynb

        * chooses 30-50% of the images in the batch at random
        * for each image in the 30-50% chosen, rotates the image by a random angle in [-60, 60]
        * returns a new array of size (129, 129) in which the rows corresponding to the 25% chosen images are the vectors corresponding to the new randomly rotated images, while the remaining rows correspond to the original images.
    Args:
        inputs: Input image batch, an array of shape (129, 129).

    Returns:
        An array of shape (129, 129) corresponding to a copy
        of the original `inputs` array that was randomly selected
        to be rotated by a random angle. The original `inputs`
        array should not be modified.
    """

    new_ims = np.zeros(inputs.shape).astype('float32')
    indices = random.randint(-1,1)
    angles = random.uniform(-30., 30.) 
    if indices == 0:
        rotate(inputs, angles, output = new_ims, order=1, reshape=False)


    return new_ims




#mask_filtered(raw, neuron_ids)  # used only when the first time of training
mask = np.load('maskfile5X.npy')
print ''
print 'mask loaded'

class NeuronSegmenDataset(Dataset):
    """Raw pixel and its label.
       Dataset splitted into 80,000 training and 20,000 validation set
    """

    def __init__(self, raw, neuron_ids, mask, phase, transform=None):
        """
          Args:
              raw(Volume): raw
              neuron_ids(Volume): neuron segmentation labels
              mask(numpy ndarray): filtered mask
              phase(String): 'train' or 'val'
              transform(callable, optional): optional data augmentation to be applied
        """

        self.phase = phase
        self.raw = raw
        self.neuron_ids = neuron_ids
        self.mask = mask
        self.transform = transform

    def __len__(self):
        """ length of the dataset """
        if self.phase == 'train':
            x = 80000
        else:
            x = 20000

        return x

    def __getitem__(self, idx):
        """
           Return 33*33 patches for each raw pixel at the center
           positive if boundary pixel, negative if non-boundary pixel
        """
        depth = self.raw.data.shape[0]
        size = self.raw.data.shape[1]


        while True:
            d = random.randint(0, depth - 1)
            h = random.randint(64, size - 65)
            w = random.randint(64, size - 65)
            ids_pixel = self.neuron_ids.data[d, h, w]
            pixel = self.mask[d, h, w]

            if idx % 2 == 0:       #control half samples to be boundary pixels
                if pixel == 100.:
                    raw_batch = self.raw.data[d][h - 64:h + 65, w - 64:w + 65].astype(
                        'float32')  # crop a 129*129 patch
                    

                    if self.transform:
                        raw_batch = self.transform(raw_batch)

                    raw_batch = raw_batch.reshape([1, 129, 129])
                    raw_batch = torch.from_numpy(raw_batch)
                    sample = (raw_batch, 0)

                    break
            elif pixel == 0.:     # the other half as non-boundary pixel
                    raw_batch = self.raw.data[d][h - 64:h + 65, w - 64:w + 65].astype(
                        'float32')  # crop 33*33 patch
                    raw_batch = raw_batch.reshape([1, 129, 129])

                    if self.transform:
                        raw_batch = self.transform(raw_batch)

                    raw_batch = torch.from_numpy(raw_batch)
                    sample = (raw_batch, 1)

                    break

        return sample



#data_transforms = {
 #   'train': transforms.Compose([
 #       transform.

  #  ]),
 #   'val': transforms.Compose({

 #   }),
#} 
batch_size = 100
emdset_seg = {x: NeuronSegmenDataset(raw, neuron_ids, mask, x, transform=random_rotation)
              for x in ['train', 'val']}
emdset_loaders = {x: DataLoader(emdset_seg[x], batch_size=batch_size, shuffle=True)
                  for x in ['train', 'val']}
dset_sizes = {x: len(emdset_seg[x]) for x in ['train', 'val']}
dset_classes = ['boundary', 'non-b']
use_gpu = torch.cuda.is_available()

print "Load num of batches: train " + str(len(emdset_loaders['train'])) + \
      '  validation ' + str(len(emdset_loaders['val']))

print ('done')
print ('')
 



def residualLayer2(conv2d1, norm2d, input, nChannels, nOutChannels=False, stride=1, conv2d2=False):
    """ Deep Residual Network
        https://github.com/gcr/torch-residual-networks   """
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





class ConvNet(nn.Module):
    def __init__(self, D_out, kernel= 3, window =2, padding=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel, padding=padding)
        self.conv2 = nn.Conv2d(32, 32, kernel, padding=padding)
        self.conv3 = nn.Conv2d(32, 64, kernel, padding=padding)
        self.conv4 = nn.Conv2d(64, 64, kernel, padding=padding)
        self.conv5 = nn.Conv2d(64, 128, kernel, padding=padding)
        self.conv6 = nn.Conv2d(128, 128, kernel, padding=padding)
        self.conv7 = nn.Conv2d(128, 256, kernel, padding=padding)
        self.conv8 = nn.Conv2d(256, 256, kernel, padding=padding)
        self.conv9 = nn.Conv2d(256, 512, kernel, padding=padding)
        self.conv10 = nn.Conv2d(512, 512, kernel, padding=padding)
        self.pool = nn.MaxPool2d(window)
        self.linear1 = nn.Linear(2*2*256, 256)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, D_out)

    def forward(self, x):


        x = F.relu(self.conv1(x))
        print "conv 1:      " + str(x.data.size())
        x = F.relu(self.conv2(x))
        print "conv 2:      " + str(x.data.size())
        x = self.pool(x)
        print "pool 1:      " + str(x.data.size())

        x = F.relu(self.conv3(x))
        print "conv 3:      " + str(x.data.size())
        x = F.relu(self.conv4(x))
        print "conv 4:      " + str(x.data.size())
        x = self.pool(x)
        print "pool 2:      " + str(x.data.size())

        x = F.relu(self.conv5(x))
        print "conv 5:      " + str(x.data.size())
        x = F.relu(self.conv6(x))
        print "conv 6:      " + str(x.data.size())
        x = self.pool(x)
        print "pool 3:      " + str(x.data.size())

        x = F.relu(self.conv7(x))
        print "conv 7:      " + str(x.data.size())
        x = F.relu(self.conv8(x))
        print "conv 8:      " + str(x.data.size())
        x = self.pool(x)
        print "pool 4:      " + str(x.data.size())

        #x = F.relu(self.conv9(x))
        # print "conv 7:      " + str(x.data.size())
        #x = F.relu(self.conv10(x))
        # print "conv 8:      " + str(x.data.size())
        #x = self.pool(x)
        # print "pool 4:      " + str(x.data.size())

        x = x.view(-1, 2*2*256)
        x = F.relu(self.linear1(x))
        #x = nn.functional.dropout(x)

        #x =  F.relu(self.linear2(x))
        #x = F.dropout(x)

        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        return x





def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
    """Decay learning rate by a factor of 0.1 every 10 epochs"""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def piecewise_scheduler(optimizer, epoch):
    if epoch % 50 ==0 :
        for param_group in optimizer.param_groups:
            lr =  param_group['lr'] / 2
            param_group['lr']  = lr

    return optimizer



def train_model (model, criterion, optimizer, lr_scheduler=None, num_epochs=100):
    since = time.time()
    train_voi_split = np.zeros(num_epochs)
    train_voi_merge = np.zeros(num_epochs)
    train_rand = np.zeros(num_epochs)

    # iterate over epoch
    for epoch in range(num_epochs):
        print ('Epoch{}/{}'.format(epoch+1, num_epochs))
        print ('-' * 10)


        # train and validation set
        for phase in ['train', 'val']:
            if phase == 'train':
                if lr_scheduler:
                    optimizer = lr_scheduler(optimizer, epoch + 1)
                model.train(True)
            else:
                model.train(True)

            running_loss = 0.
            running_accuracy = 0.
            total = 0




            # iterate over each batch
            for i, data in enumerate(emdset_loaders[phase]):
                inputs, labels = data
                if use_gpu:
                    model = model.cuda()
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


                optimizer.zero_grad()   # clean gradients in buffer

                outputs = model(inputs)
                # (batch_split, batch_merge) = voi(outputs.numpy(), labels.numpy(), ignore_groundtruth = [0])
                # batch_rand = adapted_rand(outputs.numpy(), labels.numpy())
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)


                if phase == 'train':
                    loss.backward()
                    optimizer.step()


                running_loss += loss.data[0]
                #total += labels.size(0)
                running_accuracy += (predicted == labels.data).sum()
                #if phase == 'val':
                #    print running_accuracy
                #    print total * 1.
                # train_voi_split[i] += batch_split
                # train_voi_merge[i] += batch_merge
                # train_rand[i] += batch_rand


                if (i+1) == 300:
                    #print str(i+1) + 'done'
                    print '---'

                # visualize random patches
                visualize_pred = True
                tt = visualize_pred and epoch == num_epochs-1 and phase == 'val' \
                     and i+1 == len(emdset_loaders['val'])
                if tt:
                    print('visualizing...')
                    images_so_far = 0
                    fig = plt.figure()
                    for j in [6, 15, 38, 41, 86, 99]:
                        images_so_far += 1
                        ax = fig.add_subplot(3, 2, images_so_far)
                        ax.axis('off')
                        ax.set_title('Pred: {},\n Labeled: {}'.format(dset_classes[int(predicted.cpu().numpy()[j])],
                                              dset_classes[labels.data[j]]))
                        ax.imshow(inputs.cpu().data[j].view(129,129).numpy())
                    fig.savefig('6p.png')
                    print 'done and saved to 6p.png'




            # normalize by number of batches
            running_loss /= (i + 1)
            running_accuracy = 100 * running_accuracy / dset_sizes[phase]
            # train_voi_split /= len(emdset_loader)
            # train_voi_merge /= len(emdset_loader)
            # train_rand /= len(emdset_loader)

            # print statistics
            if epoch % 1 == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, running_loss, running_accuracy
                ))
                # print "\tvoi split   : " + str(train_voi_split)
                # print "\tvoi merge   : " + str(train_voi_merge)
                # print "\tadapted RAND: " + str(train_rand)



            # print training time after each epoch
            #if phase == 'val':
	           # train_time = time.time()
	            #time_elapsed = train_time - since
	           # print('epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	           # since = train_time
            
            



    # Visualize the model. raw, labeled, predicted
    visualize = False
    if visualize:
        print('')
        print('Begin to visualize model..')
        visualize_model(model)

        time_elapsed = time.time() - train_time
        print('Visualizing complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))





def visualize_model(model, i = 80, s  = 300):
    """ Args:
        model: model
        i: depth of the raw image to visualize
        s: crop the 1250*1250 image to the size of s*s
    """


    fig = plt.figure(figsize=(15,6))

    ax_ori = fig.add_subplot(1,3,1)
    ax_lab = fig.add_subplot(1,3,2)
    ax_pred = fig.add_subplot(1,3,3)

    ax_ori.imshow(raw.data[i][0:s, 0:s])
    ax_ori.set_title('raw')
    ax_lab.imshow(neuron_ids.data[i][0:s, 0:s])
    ax_lab.set_title('labeled')


    preds = np.empty([s*s])
    for j in range(s*s):
        pixel = raw.data[i][j/s, j%s]
        input = np.random.uniform(-10000, 0, (1, 1, 33, 33)).astype('float32')  ## boundary patch: positive
        input[0, 0, 16, 16] = pixel
        input = torch.from_numpy(input)

        model.train(False)
        if use_gpu:
            model = model.cuda()
            input = Variable(input.cuda())
        else:
            input = Variable(input)



        outputs = model(input)
        _, pred = torch.max(outputs.data, 1)
        pred = pred.cpu().numpy()
        if pred[0] == 0:
            preds[j] = 20000
        else:
            preds[j] = 100

        if j == 30000:
            print '1/3 done'
        if j == 60000:
            print '2/3 done'

    print preds.reshape(s, s)

    ax_pred.imshow(preds.reshape((s,s)))
    ax_pred.set_title('predicted')


    ax_lab.axis('off')
    ax_ori.axis('off')
    ax_pred.axis('off')

    plt.show()
    fig.savefig('vi.png')
    print('saved as vi.png')








num_classes = 2
num_epochs = 100
#model = ConvNet(num_classes )
#model = DeepResNet18(num_classes)
#model = DeepResNet34(num_classes )
model = DeepResNet50(num_classes)
#model = DeepResNet101(num_classes )
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
#optimizer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10], gamma=0.5)
criterion = nn.CrossEntropyLoss()
print('')
print('START TRAINING  ...')
print(time.time())
print('ResNet50. 33% 30deg lr50')
train = train_model(model, criterion, optimizer, lr_scheduler=piecewise_scheduler, num_epochs=num_epochs)
 


