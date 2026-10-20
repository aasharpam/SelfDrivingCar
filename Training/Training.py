import os
from UtlisTraining import getName
from UtlisClaude import createCombinedModel, dataGenWithStopSigns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def importDataInfoWithStopSigns(track_path, stop_signs_path):
    """Import track data and add stop sign information"""
    # Import original track data
    columns = ['Center', 'Steering']
    data = pd.DataFrame()
    noOfFolders = len(os.listdir(track_path)) // 2

    # Import original track data with steering values
    for x in range(0, 8):
        dataNew = pd.read_csv(os.path.join(track_path, f'log_{x}.csv'), names=columns)
        print(f'{x}:{dataNew.shape[0]} ', end='')
        dataNew['Center'] = dataNew['Center'].apply(getName)
        # Add column for stop signs, initially all 0
        dataNew['StopSign'] = 0
        data = pd.concat([data, dataNew], ignore_index=True)

    # Import and add stop sign images
    stop_sign_data = []
    for img_name in os.listdir(stop_signs_path):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            stop_sign_data.append({
                'Center': f'myData/{img_name}',  # Store relative path
                'Steering': 0.0,  # Neutral steering for stop sign images
                'StopSign': 1  # Mark as containing stop sign
            })

    # Add stop sign data to main dataset
    stop_sign_df = pd.DataFrame(stop_sign_data)
    data = pd.concat([data, stop_sign_df], ignore_index=True)

    print('\nTotal Track Images:', len(data) - len(stop_sign_data))
    print('Total Stop Sign Images:', len(stop_sign_data))
    print('Total Combined Images:', len(data))

    return data


def loadDataWithStopSigns(track_path, stop_signs_base_path, data):
    """Load images and their corresponding labels"""
    imagesPath = []
    steering = []
    stop_signs = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]

        # Handle both track images and stop sign images
        if indexed_data['Center'].startswith('myData'):
            # For stop sign images
            img_path = os.path.join(stop_signs_base_path, indexed_data['Center'].split('/')[-1])
        else:
            # For track images
            img_path = os.path.join(track_path, indexed_data['Center'])

        imagesPath.append(img_path)
        steering.append(float(indexed_data['Steering']))
        stop_signs.append(float(indexed_data['StopSign']))

    return np.asarray(imagesPath), np.asarray(steering), np.asarray(stop_signs)


def balanceDataWithStopSigns(data, display=True):
    """Balance the dataset while preserving stop sign images"""
    # Separate stop sign images from regular track images
    stop_sign_data = data[data['StopSign'] == 1]
    track_data = data[data['StopSign'] == 0]

    # Balance track data as before
    nBin = 31
    samplesPerBin = 300
    hist, bins = np.histogram(track_data['Steering'], nBin)

    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(track_data['Steering']), np.max(track_data['Steering'])),
                 (samplesPerBin, samplesPerBin))
        plt.title('Data Visualization (Track Images)')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in track_data.index:
            if track_data.loc[i, 'Steering'] >= bins[j] and track_data.loc[i, 'Steering'] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)

    print('Removed Track Images:', len(removeindexList))
    track_data.drop(index=removeindexList, inplace=True)

    # Combine balanced track data with stop sign data
    balanced_data = pd.concat([track_data, stop_sign_data], ignore_index=True)
    print('Final Track Images:', len(track_data))
    print('Final Stop Sign Images:', len(stop_sign_data))
    print('Total Combined Images:', len(balanced_data))

    if display:
        # Plot final distribution of steering angles (excluding stop sign images)
        hist, _ = np.histogram(track_data['Steering'], nBin)
        plt.bar(center, hist, width=0.03)
        plt.plot((np.min(track_data['Steering']), np.max(track_data['Steering'])),
                 (samplesPerBin, samplesPerBin))
        plt.title('Balanced Data (Track Images)')
        plt.xlabel('Steering Angle')
        plt.ylabel('No of Samples')
        plt.show()

    return balanced_data

def trainModel(track_path, stop_signs_path):
    print('Loading and preparing data...')

    # Import and combine data
    data = importDataInfoWithStopSigns(track_path, stop_signs_path)

    # Balance the dataset
    data = balanceDataWithStopSigns(data, display=True)

    # Load all images and labels
    # Get the base path (parent directory) of the stop signs folder
    stop_signs_base_path = stop_signs_path

    imagesPath, steering, stop_signs = loadDataWithStopSigns(track_path, stop_signs_base_path, data)

    # Split the dataset
    (xTrain, xVal, yTrain, yVal,
     stopTrain, stopVal) = train_test_split(imagesPath, steering, stop_signs,
                                            test_size=0.2, random_state=10)

    print('Training Set:', len(xTrain))
    print('Validation Set:', len(xVal))

    # Create and train model
    model = createCombinedModel()  # Using the model architecture from previous example

    history1 = model.fit(
        dataGenWithStopSigns(xTrain, yTrain, stopTrain, 100, 1),
        steps_per_epoch=100,
        epochs=10,
        validation_data=dataGenWithStopSigns(xVal, yVal, stopVal, 50, 0),
        validation_steps=50
    )

    model.save('combined_model.h5')
    return history1


if __name__ == "__main__":
    print('Setting UP')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Define paths based on your structure
    TRACK_PATH = 'DataCollected'  # Contains IMG0, IMG1, etc., and log files
    STOP_SIGNS_PATH = 'myData'  # Contains stop sign images

    # Train model
    history = trainModel(TRACK_PATH, STOP_SIGNS_PATH)

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['steering_output_loss'])
    plt.plot(history.history['val_steering_output_loss'])
    plt.title('Steering Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['stop_sign_output_loss'])
    plt.plot(history.history['val_stop_sign_output_loss'])
    plt.title('Stop Sign Detection Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])

    plt.show()