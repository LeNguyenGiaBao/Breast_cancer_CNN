# Breast_cancer_use_CNN_Nhom9

The project using CNN architecture to predict breast cancer cell.

## Data
Data link: https://drive.google.com/drive/folders/1iSE7RTkEiTZc_Kn8f5zBMfdm6rdCwzqw?usp=sharing

Description: 
Breast cancer data have 4 types: 
- Benign
- InSitu
- Invasive
- Normal
Size: 13gb
Total images: 1624

## Model
Base model: VGG16.

Fine - turning with 4 classes in output layer  
Total params: 15,305,156

## Training

Epoch: 50  
Batch_size = 64  


![Alt Text](https://github.com/LeNguyenGiaBao/Breast_cancer_use_CNN_Nhom9/blob/master/accuracy.png)  
Accuracy graph

![Alt Text](https://github.com/LeNguyenGiaBao/Breast_cancer_use_CNN_Nhom9/blob/master/loss.png)  
Loss graph


## Test model
Run predict.py, with data_path variable is your data set path 
