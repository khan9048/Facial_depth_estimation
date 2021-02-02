# Group_C3I-Team---3D-Synthetic-Data-Projects-group
3D-Synthetic-Data:<br/>
This is the project page for our research<br/>

High-Accuracy Facial Depth Models derived from 3D Synthetic Data<br/>
https://ieeexplore.ieee.org/document/9180166<br/>

Methodology for Building Synthetic Datasets with Virtual Humans<br/>
https://ieeexplore.ieee.org/abstract/document/9180188<br/>


We will also update latest progress and available sources to this repository~ 

# Note
This repository contains PyTorch implementations of the paper 'Accurate 2D Facial Depth Models Derived from a 3D Synthetic Dataset' on our new high quality generated 3D synthetic dataset along with training on https://github.com/ialhashim/DenseDepth.

# Requirements
pip install...:
keras, Pillow, matplotlib, opencv-python, scikit-image, sklearn, pathlib, pandas, -U efficientnet, 
pip install https://www.github.com/keras-team/keras-contrib.git, torch, torchvision

# Data
The datasets are released and can be downloaded from the following link...<br/>
![Fk_CVPR_01](https://user-images.githubusercontent.com/49758542/106660724-4743eb80-6598-11eb-9387-5f6a451f53f7.png)
![da](https://user-images.githubusercontent.com/49758542/106660977-9ab63980-6598-11eb-8754-3235cfd43bf3.png)



# Testing
python test.py

# Evaluation
python eval.py

# Training
Once the dataset is ready, you can train the network using following command.<br/>
python train.py

# Modified https://www.github.com/keras-team/keras-contrib.git
open the FaceDepth directory 
# Testing
python Facedepth_test.py

This will save results into the 'pre_syn_test' folder

# Evaluation
python eval.py

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


