#!/usr/bin/env python
# coding: utf-8

# In[0]:
from keras.models import load_model
import shutil
import numpy as np
import multiprocessing
import os
import cv2
import random
import time
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from pickle_mixin import test
import tensorflow as tf
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential, load_model
from keras import layers
from keras import utils as np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization, InputLayer, Lambda, Input
from array import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as resnet_preprocess
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from keras.applications.xception import Xception, preprocess_input as xception_preprocess
from keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess
from keras.layers.merge import concatenate

# In[2]:
# 331x331 is the dimensions required by our pretrained models which we are using for transfer learning
n = 331
# This is the directory of your dataset 1 main directory + sub directories for each class
#mainDir = 'C:/Users/Hugh Mungus/Documents/VSCode/Dog Breed Scanner/Images'
trainDir = r'C:\Users\Hugh Mungus\Desktop\dogs3\train'
testDir = r'C:\Users\Hugh Mungus\Desktop\dogs3\test'
valDir = r'C:\Users\Hugh Mungus\Desktop\dogs3\val'
rawDir = r'C:\Users\Hugh Mungus\Desktop\DogDetective Dataset2'

# %% Script to move files from 'animal dog ... ' to 'dog' and remove that folder
for root, dirnames, files in os.walk(rawDir):
    for dirname in dirnames:
        if(dirname == 'train' or dirname == 'test' or dirname == 'val'):
            for root, dirnames, files in os.walk(root):
                for dirname in dirnames:
                    if(dirname != 'train' and dirname != 'test' and dirname != 'val' and dirname != 'data'):
                        tempDirName = root + '\\' + dirname
                        print('Directory name: ', tempDirName)
                        if(dirname.find('animal dog ') != -1):
                            newDirName = tempDirName.replace('animal dog ', '')

                            if(os.path.isdir(newDirName) == False):
                                os.mkdir(newDirName)

                            for root, dirnames, files, in os.walk(tempDirName):
                                for file in files:
                                    tempFileName = tempDirName + '\\' + file
                                    newFileName = newDirName + '\\' + file
                                    os.rename(tempFileName, newFileName)
                                    print(
                                        'Replaced: [', tempFileName, ']\n---> With: [', newFileName, ']')

                            os.rmdir(tempDirName)
# %%
for root, dirnames, files in os.walk(rawDir):
    for dirname in dirnames:
        if(dirname == 'train' or dirname == 'test' or dirname == 'val'):
            for root, dirnames, files in os.walk(root):
                for dirname in dirnames:
                    if(dirname != 'train' and dirname != 'test' and dirname != 'val' and dirname != 'data'):
                        tempDirName = root + '\\' + dirname
                        print('Directory name: ', tempDirName)
                        if(dirname.find('animal ') != -1):
                            newDirName = tempDirName.replace('animal ', '')

                            if(os.path.isdir(newDirName) == False):
                                os.mkdir(newDirName)

                            for root, dirnames, files, in os.walk(tempDirName):
                                for file in files:
                                    tempFileName = tempDirName + '\\' + file
                                    newFileName = newDirName + '\\' + file
                                    os.rename(tempFileName, newFileName)
                                    print(
                                        'Replaced: [', tempFileName, ']\n---> With: [', newFileName, ']')

                            os.rmdir(tempDirName)

# %% builds the train images folder
shutil.rmtree(trainDir)
os.mkdir(trainDir)
# %%
for root, dirnames, files in os.walk(rawDir):
    for dirname in dirnames:
        if(dirname == 'train'):
            print("train")
            for root, dirnames, files in os.walk(root + '\\' + dirname):
                for dirname in dirnames:
                    tempDirName = root + '\\' + dirname
                    for root, _, files in os.walk(tempDirName):

                        newDirName = trainDir + '\\' + dirname

                        if(os.path.isdir(newDirName) == False):
                            os.mkdir(newDirName)

                        for file in files:
                            tempFileName = tempDirName + '\\' + file
                            newFileName = newDirName + '\\' + file
                            if(os.path.isfile(newFileName) == False):
                                shutil.copyfile(tempFileName, newFileName)
                                print('Moved: [', tempFileName,
                                      ']\n---> To: [', newFileName, ']')

                        print(newDirName, ': ', len(files), 'files')


# %% builds the test images folder
shutil.rmtree(testDir)
os.mkdir(testDir)

# %%
for root, dirnames, files in os.walk(rawDir):
    for dirname in dirnames:
        if(dirname == 'test'):
            for root, dirnames, files in os.walk(root + '\\' + dirname):
                for dirname in dirnames:
                    tempDirName = root + '\\' + dirname
                    for root, _, files in os.walk(tempDirName):

                        newDirName = trainDir + '\\' + dirname

                        if(os.path.isdir(newDirName) == False):
                            os.mkdir(newDirName)

                        for file in files:
                            tempFileName = tempDirName + '\\' + file
                            newFileName = newDirName + '\\' + file
                            if(os.path.isfile(newFileName) == False):
                                shutil.copyfile(tempFileName, newFileName)
                                print('Moved: [', tempFileName,
                                      ']\n---> To: [', newFileName, ']')

                        print(newDirName, ': ', len(files), 'files')

# %% builds the val images folder
shutil.rmtree(valDir)
os.mkdir(valDir)

# %%
for root, dirnames, files in os.walk(rawDir):
    for dirname in dirnames:
        if(dirname == 'val'):
            for root, dirnames, files in os.walk(root + '\\' + dirname):
                for dirname in dirnames:
                    tempDirName = root + '\\' + dirname
                    for root, _, files in os.walk(tempDirName):

                        newDirName = valDir + '\\' + dirname

                        if(os.path.isdir(newDirName) == False):
                            os.mkdir(newDirName)

                        for file in files:
                            tempFileName = tempDirName + '\\' + file
                            newFileName = newDirName + '\\' + file
                            if(os.path.isfile(newFileName) == False):
                                shutil.copyfile(tempFileName, newFileName)
                                print('Moved: [', tempFileName,
                                      ']\n---> To: [', newFileName, ']')

                        print(newDirName, ': ', len(files), 'files')

# In[1]:
# Category dictionary to map our keys to category names
categoryDict = {}
train_imagenames = []
train_categories = []

val_imagenames = []
val_categories = []

test_imagenames = []
test_categories = []

print("Training: ")
count1 = 0
for dirname, _, filenames in os.walk(trainDir):
    # Just taking the last portion of the directory name which is the class name
    category = dirname[41:]
    # Skipping the first directory because it has no images
    if count1 != 0:
        for filename in filenames:
            if filename != '':
                # Making sure we stay within the right category directory
                index = dirname.find(category)
                if(index > -1):
                    # Appending the full file URI to the imagenames array
                    train_imagenames.append(dirname + '/' + filename)
                    # Count -1 because we need our keys to start at 0
                    train_categories.append(count1-1)
        # Mapping indexes 0-119 with class name strings
        categoryDict[count1-1] = category
        # Printing just the last portion of the name because first part is unique int ID
        print('Category:', category, '- Key:',
              count1-1, '- Total Imgs:', len(filenames))
    count1 += 1

print("Validation: ")
count2 = 0
for dirname, _, filenames in os.walk(valDir):
    # Just taking the last portion of the directory name which is the class name
    category = dirname[39:]
    # Skipping the first directory a.k.a the main directory because it has no images
    if count2 != 0:
        for filename in filenames:
            if filename != '':
                # Making sure we stay within the right category directory
                index = dirname.find(category)
                if(index > -1):
                    # Appending the full file URI to the imagenames array
                    val_imagenames.append(dirname + '/' + filename)
                    val_categories.append(count2-1)
                    # Count -1 because we need our keys to start at 0
        # Printing just the last portion of the name because first part is unique int ID
        print('Category:', category, '- Key:',
              count2-1, '- Total Imgs:', len(filenames))
    count2 += 1

print("Testing: ")
count3 = 0
for dirname, _, filenames in os.walk(testDir):
    # Just taking the last portion of the directory name which is the class name
    category = dirname[40:]
    # Skipping the first directory a.k.a the main directory because it has no images
    if count3 != 0:
        for filename in filenames:
            if filename != '':
                # Making sure we stay within the right category directory
                index = dirname.find(category)
                if(index > -1):
                    # Appending the full file URI to the imagenames array
                    test_imagenames.append(dirname + '/' + filename)
                    test_categories.append(count3-1)
                    # Count -1 because we need our keys to start at 0
        # Printing just the last portion of the name because first part is unique int ID
        print('Category:', category, '- Key:',
              count3-1, '- Total Imgs:', len(filenames))
    count3 += 1

# In[3]:

# In[4]:
# Creating a dataframe with our imagenames and categories
trainDF = pd.DataFrame({
    'filename': train_imagenames,
    'category': train_categories
})

testDF = pd.DataFrame({
    'filename': test_imagenames,
    'category': test_categories
})

valDF = pd.DataFrame({
    'filename': val_imagenames,
    'category': val_categories
})

print(trainDF)
print(testDF)
print(valDF)

# Pulling an array of sorted unique dog breeds from the dataframe
dogBreeds = (sorted(list(set(trainDF.category))))
num_classes = len(dogBreeds)
print("Dog Breeds: ", num_classes)

with open(r'C:\Users\Hugh Mungus\Documents\VSCode\DogDetectiveFrontend\public\dogBreeds.txt', 'w') as fp:
    for item in dogBreeds:
        # write each item on a new line
        fp.write("%s," % item)
    print('Done')

# dogBreeds[119]
# In[5]:
# Plotting the image count per category
trainDF['category'].value_counts().plot.bar()
print(trainDF['category'].value_counts())

print('\n Total images: ', trainDF.shape[0])
print(categoryDict)

a_file = open("categoryDict.pkl", "wb")
pickle.dump(categoryDict, a_file)
a_file.close()

# In[6]:
# Pretty self explanatory
print(len(trainDF))
print(len(train_categories))
print(train_imagenames[10])

# In[5]:
# Plotting the image count per category
valDF['category'].value_counts().plot.bar()
print(valDF['category'].value_counts())

print('\n Total images: ', valDF.shape[0])
print(categoryDict)

# In[6]:
# Pretty self explanatory
print(len(valDF))
print(len(val_categories))
print(val_imagenames[10])

# In[5]:
# Plotting the image count per category
testDF['category'].value_counts().plot.bar()
print(trainDF['category'].value_counts())

print('\n Total images: ', testDF.shape[0])
print(categoryDict)
# In[6]:
# Pretty self explanatory
print(len(testDF))
print(len(test_categories))
print(test_imagenames[10])


# In[7]:
# Clearing the backend to avoid memeory allocation problems
tf.keras.backend.clear_session()

# In[8]:
# Shuffling the dataframe
trainDF.sample(frac=1)
valDF.sample(frac=1)
testDF.sample(frac=1)

# In[9]:
xtrain = trainDF.filename
ytrain = trainDF.category
xtest = testDF.filename
ytest = testDF.category
xval = valDF.filename
yval = valDF.category

# In[10]:
# Creating our input shape and layer for our pretraining models
input_shape = (n, n, 3)
input_layer = Input(shape=input_shape)

# Extracting only the features from each pre-trained model
preprocessor_resnet = Lambda(resnet_preprocess)(input_layer)
inception_resnet = InceptionResNetV2(weights='imagenet',
                                     include_top=False, input_shape=input_shape, pooling='avg')(preprocessor_resnet)

preprocessor_inception = Lambda(inception_preprocess)(input_layer)
inception_v3 = InceptionV3(weights='imagenet',
                           include_top=False, input_shape=input_shape, pooling='avg')(preprocessor_inception)

preprocessor_xception = Lambda(xception_preprocess)(input_layer)
xception = Xception(weights='imagenet',
                    include_top=False, input_shape=input_shape, pooling='avg')(preprocessor_xception)

preprocessor_nasnet = Lambda(nasnet_preprocess)(input_layer)
nasnet = NASNetLarge(weights='imagenet',
                     include_top=False, input_shape=input_shape, pooling='avg')(preprocessor_nasnet)

# In[11]:
# Merging our features and creating a base model for transfer learning
mergedFeatures = concatenate(
    [inception_v3, xception, nasnet, inception_resnet])
    
model = Model(inputs=input_layer, outputs=mergedFeatures)

# In[12]:
# Printing a summary of our model - important thing to note is the output
model.summary()
model.output.shape

# In[13]:
# A function to extract features from our dataset


def feature_extractor(dataframe):
    img_size = (n, n, 3)
    data_size = (len(dataframe))
    batch_size = 8

    x = np.zeros([data_size, 9664], dtype=np.uint8)
    datagen = ImageDataGenerator()
    temp = datagen.flow_from_dataframe(dataframe,
                                       x_col='filename', class_mode=None,
                                       batch_size=8, shuffle=False, target_size=(img_size[:2]), color_mode='rgb')
    i = 0
    for input_batch in temp:
        input_batch = model.predict(input_batch)
        x[i * batch_size: (i + 1) * batch_size] = input_batch
        i += 1
        if i * batch_size >= data_size:
            break
    return x


# In[14]:
# Creating callbacks to stop training if no improvements
early_quit = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=20,
                                              restore_best_weights=True)
checkpoint = ModelCheckpoint('C:/Users/Hugh Mungus/Documents/VSCode/Dog Breed Scanner/',
                             monitor='val_loss', mode='min', save_best_only=True)
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

callbacks = [early_quit, checkpoint]

# In[15]:
temp_train = pd.DataFrame({
    'filename': xtrain,
    'category': ytrain
})
train_y = to_categorical(temp_train.category)
train_x = feature_extractor(temp_train)

# In[]:
trainDF = pd.DataFrame({
    'filename': train_imagenames,
    'category': train_categories
})
# In[];

temp_val = pd.DataFrame({
    'filename': xval,
    'category': yval
})

val_y = to_categorical(temp_val.category)
val_x = feature_extractor(temp_val)

# In[16]:
print(len(set(trainDF.category)))
print(train_y.shape[0:])

# %% save model
model.save('V2features.h5')

# In[16]: load model
model = load_model("120breedfeatures.h5")

# In[17]:
len(train_x)
# In[17]:
# Creating our final layer of dense neurons for classification
denseCNN = Sequential([
    InputLayer(train_x.shape[1:]),
    Dropout(0.7),
    Dense(num_classes, activation='softmax')
])

# In[17]:
# categorical_crossentropy for multi-classification problem
denseCNN.compile(optimizer='adam',
                 loss='categorical_crossentropy', metrics=['accuracy'])

# In[18]:
denseCNN = load_model("120breedmodel.h5")

# In[19]:
epochs = 10
# Training our model passing our x images and y categories
history = denseCNN.fit(
    train_x, train_y,
    batch_size=32,
    epochs=epochs,
    validation_data=(val_x, val_y),
    callbacks=callbacks
)

# In[]:
print(results)
# In[]
results = denseCNN.evaluate(val_x, val_y, batch_size=32)
print(results)

# In[20]:
# Plotting the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
ax1.plot(denseCNN.history['loss'], color='b', label="Training loss")
ax1.plot(denseCNN.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))
ax1.legend()

ax2.plot(denseCNN.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(denseCNN.history['val_accuracy'],
         color='r', label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))
ax2.legend()

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
# In[21]:

denseCNN.save('V2model.h5')

# In[22]: Running the testing cycle
testDF = pd.DataFrame({
    'filename': xtest,
    'category': ytest
})

test_x = feature_extractor(testDF)
test_y = to_categorical(testDF.category)

denseCNN.evaluate(test_x, test_y)

# In[24]: Random file
rand = random.randint(0, len(ytest))
img = cv2.imread(np.asarray(xtest)[rand])

res = cv2.resize(img, (n,n), interpolation=cv2.INTER_LINEAR)
print(np.asarray(xtest)[rand])
img_uri = np.asarray(xtest)[rand]
plt.imshow(res)

imDF = pd.DataFrame({
    'filename': img_uri
}, index=[0])

x_imgFeatures = feature_extractor(imDF)

y_pred = denseCNN.predict(x_imgFeatures)

results = []

for var in dogBreeds:
    results.append(categoryDict[var])

resultsDF = pd.DataFrame({
    'Breed': results,
    'Confidence': y_pred[0]
})

column = resultsDF["Confidence"]
maxIndex = column.idxmax()

guess = resultsDF.nlargest(3, "Confidence")

print(guess)

print("Best Guess: ", resultsDF.Breed[maxIndex],
      " - Confidence: ", resultsDF.Confidence[maxIndex])

# %%:
img = cv2.imread(np.asarray(xtest)[rand])

test_img_dir = r'C:\Users\Hugh Mungus\Desktop\my pics\jinx.jpg'
print(np.asarray(test_img_dir))
test_img = cv2.imread(test_img_dir)
img_uri = np.asarray(test_img_dir)

test_img = cv2.resize(test_img, (n,n), interpolation=cv2.INTER_LINEAR)
test_img = np.expand_dims(test_img, 0)

print(test_img.shape)

x_imgFeatures = model.predict(test_img)
print(x_imgFeatures.shape)
y_pred = denseCNN.predict(x_imgFeatures)

results = []

for var in dogBreeds:
    results.append(categoryDict[var])

resultsDF = pd.DataFrame({
    'Breed': results,
    'Confidence': y_pred[0]
})

column = resultsDF["Confidence"]
maxIndex = column.idxmax()
guess = resultsDF.nlargest(3, "Confidence")

print(guess)

print("Best Guess: ", resultsDF.Breed[maxIndex],
      " - Confidence: ", resultsDF.Confidence[maxIndex])
# %%:

def feature_extractor(dataframe):
    img_size = (n, n, 3)
    data_size = (len(dataframe))
    batch_size = 8
    # Important thing to note is the second value after data_size should be same as the
    # second value of your models TensorShape (i.e. [None, 9664])
    # if you use all 4 pretrained models the val will be 9664
    x = np.zeros([data_size, 9664], dtype=np.uint8)
    datagen = ImageDataGenerator()
    temp = datagen.flow_from_dataframe(dataframe,
                                       x_col='filename', class_mode=None,
                                       batch_size=8, shuffle=False, target_size=(img_size[:2]), color_mode='rgb')
    i = 0
    for input_batch in temp:
        print(input_batch.shape)
        input_batch = model.predict(input_batch)
        print(input_batch.shape)
        x[i * batch_size: (i + 1) * batch_size] = input_batch
        i += 1
        if i * batch_size >= data_size:
            break
    return x

# In[276]: Specific file
test_img_dir = r'C:\Users\Hugh Mungus\Desktop\my pics\jinx.jpg'
test_img = cv2.imread(test_img_dir)
res = cv2.resize(test_img, (n, n), interpolation=cv2.INTER_LINEAR)
plt.imshow(res)
test_img_uri = np.asarray(test_img_dir)

imDF = pd.DataFrame({
    'filename': test_img_uri
}, index=[0])

x_imgFeatures = feature_extractor(imDF)
y_pred = denseCNN.predict(x_imgFeatures)

newArray = y_pred[0]

results = []

for var in dogBreeds:
    results.append(categoryDict[var])

resultsDF = pd.DataFrame({
    'Breed': results,
    'Confidence': y_pred[0]
})

column = resultsDF["Confidence"]
maxIndex = column.idxmax()
temp = resultsDF.nlargest(3, "Confidence")

print(temp)

print("Best Guess: ", resultsDF.Breed[maxIndex],
      " - Confidence: ", resultsDF.Confidence[maxIndex])
# In[256]: Random file

# In[26]: Specific file

# In[27]:
# %%
print(xtest[10])

# %%
