# Group_C3I-Team---3D-Synthetic-Data-Projects-group
3D-Synthetic-Data:<br/>
This is the project page for our research<br/>

High-Accuracy Facial Depth Models derived from 3D Synthetic Data<br/>
https://ieeexplore.ieee.org/document/9180166<br/>

Methodology for Building Synthetic Datasets with Virtual Humans<br/>
https://ieeexplore.ieee.org/abstract/document/9180188<br/>


We will also update latest progress and available sources to this repository~ 

# Note
This repository contains PyTorch implementations of the paper 'An Efficient Encoder-Decoder Model for Portrait Depth Estimation from Single Images
trained on Pixel-Accurate Synthetic Data' on our new high quality generated 3D synthetic dataset along with training on 

https://github.com/ialhashim/DenseDepth.

https://github.com/cogaplex-bts/bts

# Requirements
pip install...:
keras, Pillow, matplotlib, opencv-python, scikit-image, sklearn, pathlib, pandas, -U efficientnet, 
pip install https://www.github.com/keras-team/keras-contrib.git, torch, torchvision

# Data
contact me on the following email: f.khan4@nuigalway.ie for the complete dataset, I will provide the download link <br/>

![data_mix](https://user-images.githubusercontent.com/49758542/106769813-49a85300-6635-11eb-9b73-dd9935f8989d.png)

![da](https://user-images.githubusercontent.com/49758542/106660977-9ab63980-6598-11eb-8754-3235cfd43bf3.png)



# Testing
python test.py

# Training
Once the dataset is ready, you can train the network using following command.<br/>
python train.py

# Modified https://www.github.com/keras-team/keras-contrib.git
download the pre-trained model and keep in the FaceDepth directory to test.

https://nuigalwayie-my.sharepoint.com/:u:/g/personal/f_khan4_nuigalway_ie/EepkuVajAhdIjZoQm5Weyx4BjXcEZy-uw5OWxxMXq1WJPA?e=rv3aSY

# Testing
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


