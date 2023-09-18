# -*- coding: utf-8 -*-
"""CarDamage"""


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf



#initialize the initial learning rate, number of epochs to train for,
# and batch size
# INIT_LR = 1e-5
# EPOCHS = 50
# BS = 64

# DIRECTORY = "/content/CarDamageDS/training"
# DIRECTORY2="/content/CarDamageDS/validation"
# CATEGORIES = ["00-damage", "01-whole"]

# data = []
# labels = []

# for category in CATEGORIES:
#     path = os.path.join(DIRECTORY, category)
#     for img in os.listdir(path):
#         img_path = os.path.join(path, img)
#         image = load_img(img_path, target_size=(224, 224))
#         image = img_to_array(image)
#         image = preprocess_input(image)

#         data.append(image)
#         labels.append(category)


# for category in CATEGORIES:
#     path = os.path.join(DIRECTORY2, category)
#     for img in os.listdir(path):
#         img_path =os.path.join(path, img)
#         image = load_img(img_path, target_size=(224, 224))
#         image = img_to_array(image)
#         image = preprocess_input(image)

#         data.append(image)
#         labels.append(category)


# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)

# data = np.array(data, dtype="float32")
# labels = np.array(labels)



# (trainX, testX, trainY, testY) = train_test_split(data, labels,
# test_size=0.20, stratify=labels, random_state=42)

# # construct the training image generator for data augmentation
# aug = ImageDataGenerator(
# rotation_range=20,
# zoom_range=0.15,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.15,
# horizontal_flip=True,
# fill_mode="nearest")

# baseModel =ResNet50(weights="imagenet", include_top=False,
#                         input_tensor=Input (shape=(224, 224, 3)))
# # construct the head of the model that will be placed on top of the
# #the base model
# headModel= baseModel.output
# headModel= MaxPooling2D(pool_size=(7, 7))(headModel)
# headModel =Flatten(name="flatten")(headModel)
# headModel =Dense(128, activation="relu")(headModel)
# headModel= Dropout(0.5)(headModel)
# headModel =Dense(64, activation="relu")(headModel)
# headModel= Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)
# # place the head FC model on top of the base model (this will become # the actual model we will train)
# model=Model(inputs=baseModel.input, outputs=headModel)

# model.summary()

# for layer in baseModel.layers:
#     layer.trainable=False
# print("[INFO] compiling model...")
# opt= Adam(learning_rate=INIT_LR)
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
# # train the head of the network
# print("[INFO] training head...")

# patience = 1
# stop_patience = 3
# factor = 0.5

# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint("damage_detection_model.h5", save_best_only=True, verbose = 0),
#     tf.keras.callbacks.EarlyStopping(patience=stop_patience, monitor='val_loss', verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, verbose=1)
# ]

# H= model.fit(aug.flow(trainX, trainY, batch_size=BS),steps_per_epoch=len(trainX) // BS,
#              validation_data=(testX, testY),
#              validation_steps=len(testX) // BS,
#              epochs=100, callbacks=callbacks, verbose=1)

class CarDamageDetection():

    def predict(self,imagePath):
        baseModel =ResNet50(weights="imagenet", include_top=False,
                        input_tensor=Input (shape=(224, 224, 3)))
        headModel= baseModel.output
        headModel= MaxPooling2D(pool_size=(7, 7))(headModel)
        headModel =Flatten(name="flatten")(headModel)
        headModel =Dense(128, activation="relu")(headModel)
        headModel= Dropout(0.5)(headModel)
        headModel =Dense(64, activation="relu")(headModel)
        headModel= Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)
        model=Model(inputs=baseModel.input, outputs=headModel)
        for layer in baseModel.layers:
            layer.trainable=False
        opt= Adam(learning_rate=1e-5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        model.load_weights('flexirideapi/ai_models/damage_detection_model.h5')
        # Load and preprocess the image
        img = load_img(imagePath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = model.predict(img_array)
        CATEGORIES = ["00-damage", "01-whole"]
        return CATEGORIES[round(np.max(prediction))];
    

        


