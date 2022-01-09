
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt

# loading mnist data ( importig only images, that is x value only for autoencoder)
(x_train, _), (x_test, _) = mnist.load_data() 

# take only first 5000 images from train and 800 images from test data mnist data set
x_train = x_train[0:10000]
x_test = x_test[0:500]


##### salt & pepper noise adding on each image  #####

import numpy as np
import random
import cv2

# creating custom function S & P noise addition
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# appying sp_noise custom function on each train & test image
x_train_noisy = []              # initialise a list for train noisy images
for image in x_train:           # selecting each images from train data set
    img = sp_noise(image,0.05)  # appying sp_noise custom function
    x_train_noisy.append(img)   # appending each images "x_train_noisy" after adding noise   
    
x_test_noisy = []
for image in x_test:
    img = sp_noise(image,0.05)
    x_test_noisy.append(img)

# converting list into numpy array for applying astype to convert the pixel value into float from int   
x_train_noisy = np.array(x_train_noisy)
x_test_noisy = np.array(x_test_noisy)

# applying normalisation and make pixel value ranges from 0 to 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train_noisy = x_train_noisy.astype('float32') / 255.
x_test_noisy = x_test_noisy.astype('float32') / 255.

# reshaping image, so that new image should be GRAY SCALE image
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), 28, 28, 1))
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 28, 28, 1))

    
####  building autoencoder model   ###

# encoder unit
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))                 # will give bottle neck out
 
# decoder unit ( transpose operation of encoder)
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

# compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# model summary
model.summary()      # the output and input image have the same shape

# fitting the model
model.fit(x_train_noisy, x_train, epochs=20, batch_size=256, shuffle=True, 
          validation_data=(x_test_noisy, x_test))


# evaluting model on test data
model.evaluate(x_test_noisy, x_test)

# save the model
model.save('denoising_autoencoder.model')

# predicted out
no_noise_img = model.predict(x_test_noisy)

# showing few example
plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")
    
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 +i+ 1)
    plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

plt.show()

# the ouput accuracy will improve as we consider more train samples instead of girst 10000 only