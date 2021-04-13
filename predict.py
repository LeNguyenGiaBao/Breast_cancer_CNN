from keras.applications import VGG16
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.models import load_model
import keras
import cv2
import numpy as np
import glob
import os


classes = ['Benign', 'InSitu', 'Invasive', 'Normal']

data_path = glob.glob(r'D:\BreastDataset\**\*.tif')     # your dataset path     

def predict_folder(image_paths, model):
    dem = 0
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        label = classes.index(label)

        img = cv2.imread(image_path)
        img = cv2.resize(img,(300, 300))

        prediction = np.argmax(model.predict(img[np.newaxis,:]))

        if (prediction != label):
            print(prediction, label, image_path)
            dem +=1
    print("Sai: ", dem, '/', len(image_paths))


def create_model(lr=1e-4,decay=1e-4/25, training=False,output_shape = len(classes)):
    vgg_model = VGG16(weights="imagenet", 
                        include_top=False,
                        input_tensor=Input(shape=(300, 300, 3)))

    ann_model = vgg_model.output
    ann_model = MaxPooling2D(pool_size=(3, 3))(ann_model)
    ann_model = Flatten(name="flatten")(ann_model)
    ann_model = Dense(128, activation="relu")(ann_model)
    ann_model = Dropout(0.5)(ann_model)
    ann_model = Dense(output_shape, activation="softmax")(ann_model)
    
    model = Model(inputs=vgg_model.input, outputs=ann_model)
        
    return model

if __name__ == "__main__":
    model = create_model()
    print(model.summary())
    model.load_weights('breast_VGG16.h5')

    predict_folder(data_path, model)
