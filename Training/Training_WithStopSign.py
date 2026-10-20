import matplotlib.image as mpimg
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

Sequential = keras.Sequential


def getName(filePath):
    return os.path.basename(filePath)


def importDataInfo(path):
    data = []
    for label in ['stop_signs', 'no_stop_signs']:
        label_path = os.path.join(path, label)
        for img in os.listdir(label_path):
            data.append({
                'Image': os.path.join(label, img),
                'Label': 1 if label == 'stop_signs' else 0
            })
    df = pd.DataFrame(data)
    print('Total Images Imported', df.shape[0])
    return df


def balanceData(data, display=True):
    n_stop_sign = len(data[data['Label'] == 1])
    n_no_stop_sign = len(data[data['Label'] == 0])
    if display:
        plt.bar(['Stop Sign', 'No Stop Sign'], [n_stop_sign, n_no_stop_sign])
        plt.title('Data Distribution')
        plt.ylabel('Number of Samples')
        plt.show()

    if n_stop_sign != n_no_stop_sign:
        min_samples = min(n_stop_sign, n_no_stop_sign)
        balanced_data = pd.concat([
            data[data['Label'] == 1].sample(min_samples),
            data[data['Label'] == 0].sample(min_samples)
        ])
        print(f'Balanced data: {len(balanced_data)} samples')
        return balanced_data
    return data


def loadData(path, data):
    imagesPath = []
    labels = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        imagesPath.append(os.path.join(path, indexed_data['Image']))
        labels.append(indexed_data['Label'])
    return np.array(imagesPath), np.array(labels)


def augmentImage(imgPath, label):
    img = mpimg.imread(imgPath)
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
    return img, label


def preProcess(img):
    img = cv2.resize(img, (224, 224))  # Resize to a standard size
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img / 255.0
    return img


@keras.saving.register_keras_serializable()
def createModel():
    model = Sequential([
        Convolution2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        Convolution2D(64, (3, 3), activation='relu'),
        Convolution2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def dataGen(imagesPath, labels, batchSize, trainFlag):
    while True:
        imgBatch = []
        labelBatch = []
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, label = augmentImage(imagesPath[index], labels[index])
            else:
                img = mpimg.imread(imagesPath[index])
                label = labels[index]
            img = preProcess(img)
            imgBatch.append(img)
            labelBatch.append(label)
        yield np.array(imgBatch), np.array(labelBatch)


# Main execution
if __name__ == "__main__":
    print('Setting UP')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # STEP 1 - INITIALIZE DATA
    path = 'myData'  # Update this path to your dataset
    data = importDataInfo(path)
    print(data.head())

    # STEP 2 - VISUALIZE AND BALANCE DATA
    data = balanceData(data, display=True)

    # STEP 3 - PREPARE FOR PROCESSING
    imagesPath, labels = loadData(path, data)

    # STEP 4 - SPLIT FOR TRAINING AND VALIDATION
    xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, labels,
                                                  test_size=0.2, random_state=10)
    print('Total Training Images: ', len(xTrain))
    print('Total Validation Images: ', len(xVal))

    # STEP 5 - CREATE MODEL
    model = createModel()

    # STEP 6 - TRAINING
    history = model.fit(dataGen(xTrain, yTrain, 32, 1),
                        steps_per_epoch=len(xTrain) // 32,
                        epochs=20,
                        validation_data=dataGen(xVal, yVal, 32, 0),
                        validation_steps=len(xVal) // 32)

    # STEP 7 - SAVE THE MODEL
    model.save('stop_sign_detector66.h5')
    print('Model Saved')

    # STEP 8 - PLOT THE RESULTS
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()