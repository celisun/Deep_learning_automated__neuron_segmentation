# CREMIchallenge2017\ Neuron Image Segmentation Task
Tentative experiments on electron microscopy images: Neuron Segmentation Task. (see at: https://cremi.org, see leaderboard at: https://cremi.org/leaderboard/)


### Experiment restuls and visualization: 
best classification results at 100 epoch:

~acc: inception: 98.68%, augmented* 98.88%; 

~loss: inception: 0.0364, augmented*: 0.0298

<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_neuron_segmentation/master/err.png" width="500">
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/acc.png" width="500">
<img src="https://github.com/celisun/cremi/blob/master/6p.png" width="600">

### Approaches:
- For this task, I trained a 2 way classifier to classify the central pixel in 127*127 sample as boudary and non-boundary. The 2-way sofmax layer was applied before the output of the network.
- Reproduced and used **residual network method**. (original: https://arxiv.org/abs/1512.03385, implementation on github: https://github.com/gcr/torch-residual-networks). This has been giving me a great boost in classificaiton results. 

It was found in the preliminary experiments (see plot below) that using a 5-7-5 window for the three conv layers in the bottleneck block of residual net (combined with a 127*127 sample size, green line) outformed the originally proposed 1-3-1 structure (gray line) by a large margin, so the experiments reported above were all trained with the 5-7-5 window.The position of batch normalization and dropout layer in the block was also changed to further optimize classification results.

<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_neuron_segmentation/master/res%20window.png" width="300">

- **Selectively choose training samples from raw (see figure below)**: the yellow area **X3 dilated boundary** pixels were avoided to be chosen, only green and purple (true boudary, background) pixels will be selected into training batches,.  
- **Random rotation techniques**: various augmentation approches were explored, including rand rotations of +/-60, rand +/- 30, on 33.33%, 50% of samples in each batc. rand +/- 60 deg on 50% of samples (see figure below) was found to perform the best so far.
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/*Filtered%20Mask.png" width="600">
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/*Visualize%20Boundary.png" width="600">
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/rot.png" width="600">

### Problems not yet solved:

- The boundary pixels from other organels (intracellular organels) should be avoided. They might be easily and falsely treated as target neuron boundaries but are actually not.
  
- the raw is originally a 3D image of size 125 * 1250 * 1250. I treated each layer in deapth 125 as an independent sample. But in later stages, the correlation between pixels at the third dimension should be addressed. 

