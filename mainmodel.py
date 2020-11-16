import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot as plt

imgdir="filepath"

# Define the parameters for the model and the number of classes
categ=int((len([i for i in os.walk(imgdir)])-1))
bsize=1200
stepsepoch=10

imgdg=ImageDataGenerator(rescale=1/255, validation_split=0.2,
                         rotation_range=0.25, vertical_flip=True)

traindg=imgdg.flow_from_directory(imgdir, target_size=(150,150), batch_size=bsize, subset="training")
valdg=imgdg.flow_from_directory(imgdir, target_size=(150,150), subset="validation")

callback1=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)
callback2=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", baseline=0.97,
                                           mode="max", patience=10)
callback3=tf.keras.callbacks.ModelCheckpoint("model.hdf5", monitor="val_accuracy",
                                             save_best_only=True)

base_model = tf.keras.applications.MobileNetV2(weights="imagenet",
                                            input_shape=(150,150,3),
                                            include_top=False)
base_model.trainable=False
inputs=tf.keras.Input(shape=(150,150,3))
x=base_model(inputs, training=False)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(units=1024, activation="relu")(x)
x=tf.keras.layers.Dropout(rate=0.2)(x)
x=tf.keras.layers.Dense(units=512, activation="relu")(x)
x=tf.keras.layers.Dropout(rate=0.2)(x)
x=tf.keras.layers.Dense(units=100, activation="relu")(x)
x=tf.keras.layers.Dense(units=100, activation="relu")(x)
x=tf.keras.layers.Dense(units=1000, name="extraction", activation="relu")(x)
outputs=tf.keras.layers.Dense(units=categ, activation="softmax")(x)
model=tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="Adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

history=model.fit_generator(traindg, steps_per_epoch=stepsepoch, epochs=5,
                    validation_data=valdg, callbacks=[callback1, callback2, callback3])
