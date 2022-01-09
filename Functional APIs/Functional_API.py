import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#loading dataset
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

#flatenning, normalisation & converting to float
x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_valid = x_valid.reshape(-1, 28*28).astype("float32")/255.0

# Functional API ( a bit more flexible than sequantial API) [[sequantial API ( very convenient but not very flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation = 'relu', name = 'first_layer')(inputs)
x = layers.Dense(256, activation = 'relu', name = 'second_layer')(x)
outputs = layers.Dense(10, activation = 'softmax')(x)
model = keras.Model(inputs = inputs, outputs = outputs)

#compile
model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer = keras.optimizers.Adam(lr = 0.001), metrics = ["accuracy"],
    )

print(model.summary())

#fit the model
model = model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_valid, y_valid))

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

# final model after applying best number of epochs= 2
model = keras.Model(inputs = inputs, outputs = outputs)
model.compile(
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer = keras.optimizers.Adam(lr = 0.001), metrics = ["accuracy"],
    )
model.fit(x_train, y_train, epochs=2, batch_size=32)

# Accuracy
print('Train Split: ')
loss, accuracy = model.evaluate(x_train, y_train, verbose = 1)
print('Accuracy : {:5.2f}'.format(accuracy))

print('Evaluation Split: ')
loss, accuracy = model.evaluate(x_valid, y_valid, verbose = 1)
print('Accuracy : {:5.2f}'.format(accuracy))


# predictions
import numpy as np
predict_valid_results = model.predict(x_valid)
predict_valid_out = np.argmax(predict_valid_results, axis = 1)

#prediction result with actual out comparison on validation data upto first 10 results
print("first 10 predicted output are:", predict_valid_out[:10])     # first 10 predicted output
print("first 10 actual output are:",y_valid[:10])                       # first 10 actual output
