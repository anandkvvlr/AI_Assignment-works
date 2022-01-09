
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import sys
import pickle
import os

#text
raw_text = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A"
print(raw_text)

# Apply tokenization and some other changes
tokenizer = Tokenizer()
tokenizer.fit_on_texts([raw_text])

# saving the tokenizer for predict function
pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([raw_text])[0]
sequence_data[:15]

len(sequence_data)
print(tokenizer.word_index)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

"""# using 3 char to predict the fourth one. so builting corresponding chara set"""

sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)

print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]

"""#differentiate each set into input and output"""

X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])

"""# converting to array"""

X = np.array(X)
y = np.array(y)

print("Data: ", X[:10])
print("Response: ", y[:10])

"""# converting class vectors into binary class metrix inorder to apply binarycross entropy as loss function later ( converting intiger value into binary  matrix representation)"""

y = to_categorical(y, num_classes=vocab_size)
y[:5]

"""#creating the model"""

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation = "relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()

"""#Plot the model"""

from tensorflow import keras
from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, to_file='plot.png', show_layer_names=True)

"""#Train the model"""

#Train the model
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(X, y, epochs=200, callbacks=[checkpoint])

"""# predict"""

from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the model and tokenizer
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):

  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))
  predicted_word = ""
  
  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
  
  print(predicted_word)
  return predicted_word

while(True):
  text = input("Enter your line: ")
  
  if text == "0":
      print("Execution completed.....")
      break
  
  else:
      try:
          text = text.split(" ")
          text = text[-1:]
          print(text)
        
          Predict_Next_Words(model, tokenizer, text)
          
      except Exception as e:
        print("Error occurred: ",e)
        continue