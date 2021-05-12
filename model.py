from IPython import get_ipython

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from skimage.transform import resize
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, History
from tensorflow.keras import backend as K
import cv2
import matplotlib.pyplot as plt

# gpu problem, please remove while running the code
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def dice_coef(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2* K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) +1
    union=K.sum(y_true_f, axis=1, keepdims=True)+ K.sum(y_pred_f, axis=1, keepdims=True)+1
    return K.mean(intersection/union)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def unet():
    inputs = Input((96,96, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocessing(imgs):
    imgs_p=[]
    for i in range(imgs.shape[0]):
        imgs_p.append(cv2.resize(imgs[i],(96,96)))
    
    imgs_p = np.array(imgs_p)
    imgs_p = imgs_p[...,np.newaxis] #to convert the image to shape (96,96,1), remove if unwanted

    return imgs_p


imgs_train=np.load('imgs_train.npy')
imgs_mask_train=np.load('imgs_mask_train.npy')


imgs_train=preprocessing(imgs_train)
imgs_mask_train=preprocessing(imgs_mask_train)

# converting the image pixel values to float datatype
imgs_train = imgs_train.astype('float32')
imgs_mask_train = imgs_mask_train.astype('float32')

# splitting the image to training and testing for prediction, remove if not required.
imgs_train0=imgs_train[:5000,:,:,:]
imgs_mask_train0=imgs_mask_train[:5000,:,:,:]

imgs_test0=imgs_train[5001:,:,:,:]
imgs_test_true=imgs_mask_train[5001:,:,:,:]

model = unet()

print(model.summary()) # output can be viewed on model_summary.txt

history=model.fit(imgs_train0, imgs_mask_train0, batch_size=32, epochs=60, verbose=1, shuffle=True,
              validation_split=0.2)


epoch=np.arange(1,61)
#output of this can be found in image directory

plt.figure(figsize=(10,8))
plt.plot(epoch,history.history['loss'], label='Training Loss')
plt.plot(epoch, history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss in dice coefficient')
plt.legend()
plt.show()
# Pretrained weights can be found on pretrained directory
model.save('model1.hdf5')
#To predict the previously segmented test image, remove if not necessary
pred_mask_imgs=model.predict(imgs_test0)
score=dice_coef(imgs_test_true,pred_mask_imgs)

print(score)




