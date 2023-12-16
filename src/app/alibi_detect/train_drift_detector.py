import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from src import RAW_DATA_DIR,DRIFT_DETECTOR_DIR
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,\
    Dense, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.saving import save_detector

def img_to_np(path, resize = True):  
    img_array = []
    fpaths = [join(path /f) for f in listdir(path) if isfile(join(path, f))]
    for fname in fpaths:
        img = Image.open(fname).convert("RGB")
        if(resize): 
            img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images

path_train = RAW_DATA_DIR / "train/images"

train = img_to_np(path_train)
train = train.astype('float32') / 255.

encoding_dim = 1024
dense_dim = [8, 8, 128]

encoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=train[0].shape),
      Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
      Flatten(),
      Dense(encoding_dim,)
  ])

decoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=(encoding_dim,)),
      Dense(np.prod(dense_dim)),
      Reshape(target_shape=dense_dim),
      Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
  ])

od = OutlierAE( threshold = 0.001,
                encoder_net=encoder_net,
                decoder_net=decoder_net)

adam = tf.keras.optimizers.Adam(learning_rate=1e-4)

od.fit(train, epochs=20, verbose=True,
       optimizer = adam)
     
save_detector(od, DRIFT_DETECTOR_DIR)