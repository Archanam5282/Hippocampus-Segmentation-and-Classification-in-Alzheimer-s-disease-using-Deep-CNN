# Hippocampus-Segmentation-and-Classification-in-Alzheimer-s-disease-using-Deep-CNN
This file includes the description of the project and intruction to run the codes. 

Objective: To build a deep learning framework with CNN models that performs hippocampal segmentation and disease classification in Alzheimer’s disease

Alzheimer’s disease (AD) is a progressive and irreversible brain degenerative disorder characterized by memory loss and cognitive impairment. At present, there are more than 90 million patients diagnosed with AD. By 2050, it is estimated that the number of AD patients will reach 300 million. No effective treatments are available to cure AD at present, while current AD treatments and medicines can only ease symptoms of AD or slow down AD progression. Therefore, the diagnoses of AD at early stage is an effective way to prevent AD. The current clinical protocol to detect volumetric changes in the hippocampus is manual segmentation, which is time-consuming, observer-dependent and challenging. As A Result, an automated approach to hippocampus segmentation is imperative to improve the efficiency and accuracy of the diagnostic workflow. Several automatic and semi-automatic segmentation approaches have been proposed, which utilize T1-weighted structural MRIs, to segment the hippocampus. 
Recently, deep learning networks, including convolutional neural networks (CNNs), have been widely used in image classification and computer vision. The deep 3D CNNs are used to extract the features of 3D medical images for classification. Deep learning has been increasingly applied in the field of biomedical image-based diagnosis.
U-net is a neural network is widely used for biomedical image segmentation. It was first designed and applied in 2015 to process biomedical images. As a general CNN focuses its task on image classification, where input is an image and output is one label, but in biomedical cases, it requires us not only to distinguish whether there is a disease, but also to localize the area of abnormality. U-Net is dedicated to solving this problem. The reason it can localize and distinguish borders is by doing classification on every pixel, so the input and output share the same size.
Our network is based on the 3D U-Net, with dilated convolutions in the lowest layer between the encoder and decoder paths, residual connections between the convolution blocks in the encoder path, and residual connections between the convolution blocks and the final layer of the decoder path. A network is presented in methodology section of this paper. The aim of this project is to evaluate the performance of the network using the hippocampus dataset provided as part of the Medical Segmentation Decathlon challenge1 hosted at MICCAI 2018, and compared it to different results of other research papers mentioned in the review of literature session.


## Data Acquisition: 
Data is acquired from http://medicaldecathlon.com/. This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images are T2 MRI scans of the full brain. In this dataset the cropped volumes are used where only the region around the hippocampus has been cut out. As the data provided was already truncated to the region of interest around the hippocampus, not much data pre-processing was required. There are total 260 images and 260 labels used. Each image is represented by a single file and has a corresponding label file which is named the same as the image file. 

## The Programming Environment:
If you would like to run the project locally, you would need a Python 3.7+ environment with the following libraries for the first two sections of the project:

numpy
nibabel
matplotlib
PIL
pydicom
json
torch (preferably with CUDA)
tensorboard

## Description of the code
There are 2 codes :
1)code.py 
2)code.ipynb 

ipynb file is used to see and understand the code as ipynb is easy to read than the py file. So .py file is the file that is used to get the results. Both the files contain same lines of code.

The .py file contains the below definitions : 


### Loading and preprocessing:
mpl_image_grid
log_to_tensorboard
save_numpy_as_image
med_reshape
LoadHippocampusData
SlicesDataset

### Evaluation methods
Dice3d
Jaccard3d
sensitivity

### U-net
UNet
UnetSkipConnectionBlock
UNetInferenceAgent
single_volume_inference_unpadded
single_volume_inference
UNetExperiment















