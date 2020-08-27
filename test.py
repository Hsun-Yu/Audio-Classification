# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Audio Classification use CNN
# In this notebook I will use all features and train a cnn model to classify audio.
# 
# ## Get features
# Look the following code the code is same as `get_audio_features` notebook

# %%
import numpy as np
max_pad_len = 216
def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=120)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None
    return mfccs

# %% [markdown]
# ## Read dataset and get all features of all audios
# 1. Read the csv file
# 
# loop
# 
# 2. get and save features of the audio of the row
# 
# 3. save the label
# 
# Look the following code:

# %%
# Load various imports 
import pandas as pd
import os
import librosa

# Set the path to the full UrbanSound dataset 
fulldatasetpath = './ESC-50/audio/'

metadata = pd.read_csv('./ESC-50/meta/meta1.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),str(row["filename"]))
    class_label = row["category"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# %% [markdown]
# ## Convert the data and labels and split the dataset
# I will use sklearn.preprocessing.LabelEncoder to encode the categorical text data into model-understandable numerical data.
# Here I will use sklearn.model_selection.train_test_split to split the dataset into training and testing sets. The testing set size will be 20% and I will set a random state.

# %%
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

# %% [markdown]
# ## Build CNN model
# Look the following code

# %%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

a, num_rows, num_columns = X.shape
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=4, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(AveragePooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(AveragePooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(AveragePooling2D(pool_size=2))

model.add(Conv2D(filters=256, kernel_size=2, activation='relu'))
model.add(AveragePooling2D(pool_size=2))

# model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.6))

model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.6))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.6))

# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.6))

model.add(Dense(num_labels, activation='softmax'))


# %%
# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# %%
# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

# %% [markdown]
# ## Train and save model

# %%
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 1000
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='save_models/weights.best.basic_cnn11.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# %% [markdown]
# ## Evalue training data and testing data

# %%
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

# %% [markdown]
# ## Observations
# The accuracy from (DNN version)0.48 to (CNN version)0.67

