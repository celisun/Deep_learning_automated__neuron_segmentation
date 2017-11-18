# CREMIchallenge2017\ Neuron Image Segmentation Task
Tentative experiments on electron microscopy images: Neuron Segmentation Task. (see at: https://cremi.org, see leaderboard at: https://cremi.org/leaderboard/)


### Experiment restuls and visualization: 
classification results at 100 epoch:
~acc inception: 98.68%, augmented* 98.88%;  ~loss inception: 0.0364 augmented*: 0.0298

<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/loss.png" width="500">
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/acc.png" width="500">
<img src="https://github.com/celisun/cremi/blob/master/6p.png" width="600">

### Approaches:
- In this tentative experiments, I treated the segmentation task as a boundary/nonboundary classification task, using 2-way sofmax for the output of my network.
- Reproduced and used residual network method. (original: https://arxiv.org/abs/1512.03385, implementation on github: https://github.com/gcr/torch-residual-networks). This has been giving me a great boost in classificaiton results.
- Sample selection from raw: only green and purple area will be selected for training batches, dilated boundary yellow area will be avoided.  
- Different random rotation techniques. In my experiment, rand+/-60 to 50% of samples in each batch performs the best.
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/*Filtered%20Mask.png" width="600">
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/*Visualize%20Boundary.png" width="600">
<img src="https://raw.githubusercontent.com/celisun/CREMIchallenge2017_segmentation_task/master/rot.png" width="600">

### Problems not yet solved:

- The boundary pixels from other organels in neurons should be avoided. They might be easily and falsely treated as target neuron boundaries but are actually not.
  
- the raw is originally a 3D image of size 125 * 1250 * 1250. I treated each layer in deapth 125 as an independent sample. But in later stages, the correlation between pixels at the third dimension should be addressed. 

