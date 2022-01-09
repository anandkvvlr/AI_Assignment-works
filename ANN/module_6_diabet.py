#tensorflow and keras are 2 packages for neural n/w model
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# print tensorflow (tf) version to make sure you have at least tensorflow 2.1.0
print(f"tensorflow version: {tf.version.VERSION}")

#import NN layer and other components.
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt # plotting data
import numpy as np # for match and arrays
import pandas as pd  # data from for the data
import seaborn as sns # for plotting

# load data set
df = pd.read_csv('F:/AI assignment/AI assignment/Module 6/A6_Datasets/diabetes.csv')

print(f'Number of rows & columns in data set:{df.shape}')

# summary of data set information
df.info()

# CHECKING NULL VALUES
df.isna().sum()  
df = df.dropna() # drop na

# split the dataset into train and test( 60% and 40%)
train_dataset, temp_test_dataset = train_test_split(df, test_size = 0.4, random_state = 1)
print(train_dataset.shape)
print(temp_test_dataset.shape)

# split test dataset into 50% test and 50% validation
test_dataset, valid_dataset = train_test_split(temp_test_dataset, test_size = 0.5)
print(test_dataset.shape)
print(valid_dataset.shape)

# statistics on the train dataset to make sure it is in good shae
train_stats = train_dataset.describe()
train_stats.pop("Outcome")
train_stats = train_stats.transpose()
train_stats

#class from each dataset assigning on new variable
train_labels = train_dataset.pop('Outcome')
test_labels = test_dataset.pop('Outcome')
valid_labels = valid_dataset.pop('Outcome')

# normalization
def norm(x):
    return(x- train_stats['mean'])/ train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_valid_data = norm(valid_dataset)

# sample of normalized data
normed_train_data.head(10)

# checkng data type float
normed_train_data.info()   # all are float

#converting the inputs to float type
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
y_valid = np.asarray(valid_labels).astype('float32')

# change input dataset name
x_train = normed_train_data
x_test = normed_test_data  
x_valid = normed_valid_data 

#defining the model
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)))#converting the int data to tuple
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # binary output (sigmoid)


#compiling model
from keras import optimizers
from tensorflow.keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])

#fit the model
model = model.fit(x_train,y_train,epochs=50,batch_size=16,validation_data=(x_valid, y_valid))

history_dict = model.history
history_dict.keys()

#Plotting validation scores
import matplotlib.pyplot as plt
acc = model.history['accuracy']
val_acc = model.history['val_accuracy']
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs= range(1,len(acc)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
'''The attribute Loc in legend() is used to specify the location of the legend.Default value of loc is loc=”best” (upper left)'''
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#fine tuning the model
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)))#converting the int data to tuple
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=16)

# Accuracy
print('Train Split: ')
loss, accuracy = model.evaluate(x_train, y_train, verbose = 1)
print('Accuracy : {:5.2f}'.format(accuracy))

print('Evaluation Split: ')
loss, accuracy = model.evaluate(x_valid, y_valid, verbose = 1)
print('Accuracy : {:5.2f}'.format(accuracy))

print('Test Split: ')
loss, accuracy = model.evaluate(x_test, y_test, verbose = 1)
print('Accuracy : {:5.2f}'.format(accuracy))

# confussion matrix
#import seaborn as sns
from sklearn.metrics import confusion_matrix

ax= plt.subplot()
predict_results = model.predict(x_test)
predict_results = (predict_results > 0.5)

cm = confusion_matrix(y_test, predict_results)
cm
sns.heatmap(cm, annot = True, ax = ax); #annot=True to annotate cells
cm

#labels, title and ticks
ax.set_xlabel('predicted labels'); ax.set_ylabel('True labels');
ax.set_title('Confussion matrix');
ax.xaxis.set_ticklabels(['positive','negative']); ax.yaxis.set_ticklabels(['positive','negative']);



