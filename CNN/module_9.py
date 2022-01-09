## Augmentation ##
#Image shifts via the width_shift_range and height_shift_range arguments.
#Image flips via the horizontal_flip and vertical_flip arguments.
#Image rotations via the rotation_range argument
#Image brightness via the brightness_range argument.
#Image zoom via the zoom_range argument.


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=40,          # Random rotation between 0 and 40
        width_shift_range=0.2,      # % shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')        # can also try nearest, constant, reflect, wrap



##############   Loading a single image and do the augmentation  ##############

#Using flow method to augment the image
# Loading a sample image  
#Can use any library to read images but they need to be in an array form
#If using keras load_img convert it to an array first

img = load_img('F:/AI assignment/AI assignment/Convolutional Neural Network_Assign_module_9/Augment_images/images/000001.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (500, 353, 3)

# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 500, 353, 3)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `augmented_output/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='F:/AI assignment/AI assignment/Convolutional Neural Network_Assign_module_9/Augment_images/augmented_output', save_prefix='man_with_puppy', save_format='jpeg'):
    i += 1
    if i > 4:
        break  # otherwise the generator would loop indefinitely
        
        

#######################  Multiple images  ######################

#Manually read each image and create an array to be supplied to datagen via flow method
dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = 'F:/AI assignment/AI assignment/Convolutional Neural Network_Assign_module_9/Augment_images/images/'
SIZE = 400
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE,SIZE))
        dataset.append(np.array(image))

x = np.array(dataset)     # this is a Numpy array with shape (7, 400, 400, 3)

i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='F:/AI assignment/AI assignment/Convolutional Neural Network_Assign_module_9/Augment_images/augmented_output', save_prefix='augments', save_format='jpeg'):
    i += 1
    if i > 27:
        break  # otherwise the generator would loop indefinitely
  

###################### accessing image in Multiclass  problem #####################
# Read directly from the folder structure using flow_from_directory

i = 0
for batch in datagen.flow_from_directory(directory='F:/AI assignment/AI assignment/Convolutional Neural Network_Assign_module_9/Augment_images/', 
                                         batch_size=16,  
                                         target_size=(400, 400),
                                         color_mode="rgb",
                                         save_to_dir='F:/AI assignment/AI assignment/Convolutional Neural Network_Assign_module_9/Augment_images/augmented_output', 
                                         save_prefix='augments', 
                                         save_format='png'):
    i += 1
    if i > 4:
        break 

#Creates 32 images for each class. 
        
#Once data is augmented, you can use it to fit a model via: fit.generator
#instead of fit()
#model = 
#fit model on augmented data
#model.fit_generator(datagen.flow(x))