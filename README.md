## Group_C3I-Team---3D-Synthetic-Data-Projects-group
* [Modeling of the Synthetic Dataset: Pixel-accurate facial depth generation process](#general-info)
* [Evaluating State-of-Art models for Single Image Depth Estimation](#technologies)
* [An Encoder-decoder based Facial Depth Estimation Model](#setup)
* [Hybrid Loss Function](#setup1)

## Single Frame Image Facial Depth Estimation
This is the project page for our research:<br/>

An Efficient Encoder-Decoder Model for Portrait Depth Estimation
from Single Images trained on Pixel-Accurate Synthetic Data

High-Accuracy Facial Depth Models derived from 3D Synthetic Data<br/>
https://ieeexplore.ieee.org/document/9180166<br/>

Methodology for Building Synthetic Datasets with Virtual Humans<br/>
https://ieeexplore.ieee.org/abstract/document/9180188<br/>

Learning 3D Head Pose From Synthetic Data: A Semi-Supervised Approach<br/>
https://ieeexplore.ieee.org/abstract/document/9369299<br/>


We will also update latest progress and available sources to this repository~ 
	
## Note
This repository contains PyTorch implementations of FaceDepth, UNet-Simple, BTS, DenseDepth.
	
## Setup & Requirements
To run this project, install it locally using pip install...:

```
$ pip install keras, Pillow, matplotlib, opencv-python, scikit-image, sklearn, pathlib, pandas, -U efficientnet,
$ pip install https://www.github.com/keras-team/keras-contrib.git, torch, torchvision
```

```
$ Python >= 3.6
Pytorch >= 1.6.0
Ubuntu 16.04
CUDA 9.2
cuDNN (if CUDA available)
```
## Pretrained model

download the pre-trained model and keep in the FaceDepth directory:

https://nuigalwayie-my.sharepoint.com/:u:/g/personal/f_khan4_nuigalway_ie/EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY

## Prepare Dataset for training & Testing 

We prepared the dataset for training and testing<br/>
contact me on the following email: f.khan4@nuigalway.ie for the complete dataset, I will provide the download link <br/>

#### Random sample frames with high-resolutions RGB images and their corresponding ground truth depth with differentvariations<br/>
![data_mix](https://user-images.githubusercontent.com/49758542/106769813-49a85300-6635-11eb-9b73-dd9935f8989d.png)

#### Samples from the generated synthetic data with different variation of head pose
![da](https://user-images.githubusercontent.com/49758542/106660977-9ab63980-6598-11eb-8754-3235cfd43bf3.png)

## Live Demo
We attach live 3d demo implementations for Pytorch. \
Sample usage for PyTorch:
```
$ cd ~/workspace/bts/pytorch
$ python bts_live_3d.py --model_name bts_nyu_v2_pytorch_densenet161 \
--encoder densenet161_bts \
--checkpoint_path ./models/bts_nyu_v2_pytorch_densenet161/model \
--max_depth 10 \
--input_height 480 \
--input_width 640
```




# Testing
python test.py

# Training
Once the dataset is ready, you can train the network using following command.<br/>
python train.py

# Modified https://www.github.com/keras-team/keras-contrib.git

# Testing

download the pre-trained model and keep in the FaceDepth directory to test.

https://nuigalwayie-my.sharepoint.com/:u:/g/personal/f_khan4_nuigalway_ie/EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY

python Facedepth_test.py

![FK_CVPR_03](https://user-images.githubusercontent.com/49758542/106770165-99871a00-6635-11eb-98f8-b7cab4fe6938.png)

![Fk_CVPR_04](https://user-images.githubusercontent.com/49758542/106661542-39429a80-6599-11eb-8efa-519a39d7628e.png)

This will save results into the 'pre_syn_test' folder

# Training
Once the dataset is ready, you can train the network using following command.<br/>
python train.py

# Citation
If you find the code, models, or data useful, please cite this paper:<br/>
F. Khan, S. Basak, H. Javidnia, M. Schukat and P. Corcoran, "High-Accuracy Facial Depth Models derived from 3D Synthetic Data," 2020 31st Irish Signals and Systems Conference (ISSC), Letterkenny, Ireland, 2020, pp. 1-5, doi: 10.1109/ISSC49989.2020.9180166.<br/>

S. Basak, H. Javidnia, F. Khan, R. McDonnell and M. Schukat, "Methodology for Building Synthetic Datasets with Virtual Humans," 2020 31st Irish Signals and Systems Conference (ISSC), Letterkenny, Ireland, 2020, pp. 1-6, doi: 10.1109/ISSC49989.2020.9180188.<br/>

@article{alhashim2018high,
  title={High quality monocular depth estimation via transfer learning},
  author={Alhashim, Ibraheem and Wonka, Peter},
  journal={arXiv preprint arXiv:1812.11941},
  year={2018}
}


