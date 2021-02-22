# %% [markdown]
# ## Importing Libraries:

# %% [code]
# #############################################################################
# Explore data:
# #############################################################################
print("\nLOADING IMG PROCESSING LIBRARIES, WAIT ...\n")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import imageio
import cv2

from keras.utils import np_utils

print("\n... LOADING COMPLETE ...\n")


print("\nLOADING TRAINING LIBRARIES, WAIT ...\n")

import tensorflow as tf
from keras import layers, Model
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3

import cv2

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, plot_confusion_matrix

import seaborn as sn

print("\n... LOADING COMPLETE ...\n")

# %% [markdown]
# ## Load Data:

# %% [code]
# Load data from files
X_train = np.load("../input/vinbigdata-chest-x-ray-256x225-original-ratio/X_train_file_256x225.npy")
Y_train = np.load("../input/vinbigdata-chest-x-ray-256x225-original-ratio/Y_train_file_256x225.npy")

# Load data from files
#X_train = np.load("../input/vinbigdata-chest-x-ray-256x256-nparray/X_train_file_256x256.npy")
#Y_train = np.load("../input/vinbigdata-chest-x-ray-256x256-nparray/Y_train_file_256x256.npy")

# %% [markdown]
# ## Data Perparation:

# %% [markdown]
# ### For binaty classifier:

# %% [code]
Y_modefied_train_data = []
X_modefied_train_data = []
for i in range(len(Y_train)):
    if Y_train[i] == 14:
        X_modefied_train_data.append(X_train[i])
        Y_modefied_train_data.append(0)
    elif Y_train[i] != 14:
        X_modefied_train_data.append(X_train[i])
        Y_modefied_train_data.append(1)
X_train = np.array(X_modefied_train_data)
Y_train = Y_modefied_train_data

del Y_modefied_train_data
del X_modefied_train_data

# %% [code]
# Reshape to be [samples][width][height][channels]
X_train_data = X_train[:10000].reshape((X_train[:10000].shape[0], 256, 225, 1)).astype('float64')
X_validate_data = X_train[10000:12500].reshape((X_train[10000:12500].shape[0], 256, 225, 1)).astype('float64')
X_test_data = X_train[12500:15000].reshape((X_train[12500:15000].shape[0], 256, 225, 1)).astype('float64')
del X_train

# Normalize inputs from 0-65536 to 0-1
X_train_data = X_train_data / 65536
X_validate_data = X_validate_data / 65536
X_test_data = X_test_data / 65536

# One hot encode outputs
Y_train_data = np_utils.to_categorical(Y_train[:10000])
Y_validate_data = np_utils.to_categorical(Y_train[10000:12500])
Y_test_data = np_utils.to_categorical(Y_train[12500:15000])
del Y_train

num_classes = Y_validate_data.shape[1]

# %% [markdown]
# ### Multi Categorical Classifier:

# %% [code]
# Reshape to be [samples][width][height][channels]
X_train_data = X_train[:10000].reshape((X_train[:10000].shape[0], 256, 225, 1)).astype('float64')
X_validate_data = X_train[10000:12500].reshape((X_train[10000:12500].shape[0], 256, 225, 1)).astype('float64')
X_test_data = X_train[12500:15000].reshape((X_train[12500:15000].shape[0], 256, 225, 1)).astype('float64')
del X_train

# Normalize inputs from 0-65536 to 0-1
X_train_data = X_train_data / 65536
X_validate_data = X_validate_data / 65536
X_test_data = X_test_data / 65536

# ------------------------------------
w = 0
Y_modefied_train_data = []
X_modefied_train_data = []
for i in range(len(Y_train[:10000])):
    if Y_train[i] == 14 and w <= 800:
        X_modefied_train_data.append(X_train_data[i])
        Y_modefied_train_data.append(Y_train[i])
        w += 1
    elif Y_train[i] != 14:
        X_modefied_train_data.append(X_train_data[i])
        Y_modefied_train_data.append(Y_train[i])
X_train_data = np.array(X_modefied_train_data)
Y_train_data = np_utils.to_categorical(Y_modefied_train_data)

w = 0
Y_modefied_validate_data = []
X_modefied_validate_data = []
for i in range(len(Y_train[10000:12500])):
    if Y_train[10000+i] == 14 and w <= 250:
        X_modefied_validate_data.append(X_validate_data[i])
        Y_modefied_validate_data.append(Y_train[10000+i])
        w += 1
    elif Y_train[10000+i] != 14:
        X_modefied_validate_data.append(X_validate_data[i])
        Y_modefied_validate_data.append(Y_train[10000+i])
X_validate_data = np.array(X_modefied_validate_data)
Y_validate_data = np_utils.to_categorical(Y_modefied_validate_data)
# -------------------------------------

# One hot encode outputs
#Y_train_data = np_utils.to_categorical(Y_train[:10000])
#Y_validate_data = np_utils.to_categorical(Y_train[10000:12500])
Y_test_data = np_utils.to_categorical(Y_train[12500:15000])
del Y_train

num_classes = Y_validate_data.shape[1]

# %% [markdown]
# ## Data Visualization:

# %% [code]
def countData(data):
    classes_distribution = {"No Finding": 0, "Abnormal": 0}

    for y in np.argmax(data, axis=-1):
        if y == 0:
            classes_distribution["No Finding"] += 1
        if y == 1:
            classes_distribution["Abnormal"] += 1
    
    return classes_distribution

y_dist_train_data = countData(Y_train_data)
y_dist_validate_data = countData(Y_validate_data)
y_dist_test_data = countData(Y_test_data)

# %% [code]
def countData(data):
    classes_distribution = {"No finding": 0, "Aortic enlargement": 0, "Cardiomegaly": 0,
           "Pleural thickening": 0, "Pulmonary fibrosis": 0, "Nodule/Mass": 0,
           "Lung Opacity": 0, "Pleural effusion": 0, "Other lesion": 0,
           "Infiltration": 0, "ILD": 0, "Calcification": 0,
           "Consolidation": 0, "Atelectasis": 0, "Pneumothorax": 0}

    for y in np.argmax(data, axis=-1):
        if y == 0:
            classes_distribution["Aortic enlargement"] += 1
        if y == 1:
            classes_distribution["Atelectasis"] += 1
        if y == 2:
            classes_distribution["Calcification"] += 1
        if y == 3:
            classes_distribution["Cardiomegaly"] += 1
        if y == 4:
            classes_distribution["Consolidation"] += 1
        if y == 5:
            classes_distribution["ILD"] += 1
        if y == 6:
            classes_distribution["Infiltration"] += 1
        if y == 7:
            classes_distribution["Lung Opacity"] += 1
        if y == 8:
            classes_distribution["Nodule/Mass"] += 1
        if y == 9:
            classes_distribution["Other lesion"] += 1
        if y == 10:
            classes_distribution["Pleural effusion"] += 1
        if y == 11:
            classes_distribution["Pleural thickening"] += 1
        if y == 12:
            classes_distribution["Pneumothorax"] += 1
        if y == 13:
            classes_distribution["Pulmonary fibrosis"] += 1
        if y == 14:
            classes_distribution["No finding"] += 1
    
    return classes_distribution

y_dist_train_data = countData(Y_train_data)
y_dist_validate_data = countData(Y_validate_data)
y_dist_test_data = countData(Y_test_data)

# %% [code]
# ========================= Data exploring ========================
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
fig.suptitle("Distribution of the labels", fontsize=20)
axs[0].bar(y_dist_train_data.keys(), y_dist_train_data.values())
axs[0].tick_params("x", labelrotation=90)
axs[1].bar(y_dist_validate_data.keys(), y_dist_validate_data.values())
axs[1].tick_params("x", labelrotation=90)
axs[2].bar(y_dist_test_data.keys(), y_dist_test_data.values())
axs[2].tick_params("x", labelrotation=90)

plt.show()

# %% [code]
# --------------- Represent sample of the 15 classes:
classes = ["Aortic enlargement", "Atelectasis", "Calcification",
           "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
           "Lung Opacity", "Nodule/Mass", "Other lesion", "Pleural effusion",
           "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "No finding"
          ]
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(15, 10))
fig.subplots_adjust(hspace=0.3, wspace=0.1)
axs = axs.ravel()
for classid in range(15):
    for i, y in enumerate(Y_train):
        if classid == y:
            image = X_train[i]
            axs[classid].axis("off")
            axs[classid].imshow(image, cmap="gray")
            axs[classid].set_title(classes[y])            
plt.show()

# %% [markdown]
# ## CNN Models:

# %% [markdown]
# ### Binarry classifier:

# %% [code]
# define the larger model
def CNN_AdvancedModel():
    # create model
    model = Sequential()
    model.add(Conv2D(15, (2, 2), input_shape=(256, 225, 1), activation='relu')) # Conv layer
    model.add(Conv2D(20, (3, 3), input_shape=(256, 225, 1), activation='relu')) # Conv layer
    model.add(Conv2D(45, (5, 5), input_shape=(256, 225, 1), activation='relu')) # Conv layer
    model.add(MaxPooling2D((2, 2))) # Max pooling layer
    model.add(Conv2D(64, (7, 7), activation='relu'))
    model.add(MaxPooling2D((2, 2))) # Max pooling layer
    model.add(Conv2D(64, (9, 9), activation='relu'))
    model.add(MaxPooling2D((2, 2))) # Max pooling layer
    model.add(Dropout(0.2))
    model.add(Flatten()) # Fully connected layer = conv operation with a 1x1 output kernel
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
  
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = CNN_AdvancedModel()
EPOCHS = 30
# Fit the model
history = model.fit(X_train_data, Y_train_data,
                    validation_data=(X_validate_data, Y_validate_data),
                    epochs=EPOCHS, batch_size=50)

#Save trained model
model.save('partly_trained.h5')

# %% [markdown]
# ### Multi Categorical Classifier:

# %% [code]
# define the larger model
def CNN_AdvancedModel():
    # create model
    model = Sequential()
    model.add(Conv2D(256, (2, 2), input_shape=(256, 225, 1), activation='relu')) # Conv layer
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2))) # Max pooling layer
    model.add(Conv2D(128, (7, 7), activation='relu'))
    model.add(MaxPooling2D((2, 2))) # Max pooling layer
    model.add(Conv2D(64, (9, 9), activation='relu'))
    model.add(MaxPooling2D((2, 2))) # Max pooling layer
    model.add(Dropout(0.2))
    model.add(Flatten()) # Fully connected layer = conv operation with a 1x1 output kernel
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
  
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = CNN_AdvancedModel()
EPOCHS = 50
# Fit the model
history = model.fit(X_train_data, Y_train_data,
                    validation_data=(X_validate_data, Y_validate_data),
                    epochs=EPOCHS, batch_size=50)

#Save trained model
model.save('partly_trained.h5')

# %% [markdown]
# ## Optimize the Space

# %% [code]
del X_train_data
del Y_train_data
del X_validate_data
del Y_validate_data

final_model = model
del model

# %% [markdown]
# ## Visualize training results:

# %% [code]
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [code]
# Final evaluation of the model
scores = final_model.evaluate(X_test_data, Y_test_data, verbose=0)
print("CNN Error: {0:.2f}%".format(100-scores[1]*100))

# %% [code]
final_model.summary()

# %% [code]
tf.keras.utils.plot_model(
    final_model,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
)

# %% [code]
Y_predict = np.argmax(final_model.predict(X_test_data), axis=-1) # Set "axis=-1" means, extract largest indices in each row.

# %% [code]
classes_names = ["No finding", "Abnormale"]

print(classification_report(np.argmax(Y_test_data, axis=-1), Y_predict, target_names=classes_names))
print(confusion_matrix(np.argmax(Y_test_data, axis=-1), Y_predict))

# %% [code]
classes_names = ["Aortic enlargement", "Atelectasis", "Calcification",
                 "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
                 "Lung Opacity", "Nodule/Mass", "Other lesion",
                 "Pleural effusion","Pleural thickening",
                 "Pneumothorax", "Pulmonary fibrosis", "No finding"
                ]

print(classification_report(np.argmax(Y_test_data, axis=-1), Y_predict, target_names=classes_names))
print(confusion_matrix(np.argmax(Y_test_data, axis=-1), Y_predict))

# %% [code]
import matplotlib.pyplot as plt
import numpy as np

def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    #ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 4)]:
        t = line.strip().split()
        if len(t) < 2: continue
        class_name = ''
        for elem in t:
            try:
                float(elem)
            except:
                class_name += elem + " "
        classes.append(class_name)
        v = [float(x) for x in t[len(class_name.strip().split()): len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(class_name)
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def main():
    sampleClassificationReport = classification_report(np.argmax(Y_test_data, axis=-1), Y_predict, target_names=classes_names)
    plot_classification_report(sampleClassificationReport)
    plt.show()
    plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

# %% [code]
df_cm = pd.DataFrame(confusion_matrix(np.argmax(Y_test_data, axis=-1), Y_predict).tolist(), index=classes_names, columns=classes_names)
plt.figure(figsize = (20, 20))
sn.heatmap(df_cm, annot=True)
