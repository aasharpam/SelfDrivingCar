# Autonomous Vehicle Project Guide

To have the car up and running, you first need to buy all the parts and also 3D print some of them if you have a 3D printer. If not, explore some hobby kit stores to find them.

A detailed Bill of Materials (BOM) is available on my website: [Project: Autonomous Vehicle](https://www.aasharpam.com/project-autonomous-vehicle.html)

## References

Here are some other sources I referred to. I am thankful to their creators for putting these resources online:

- [YouTube video](https://www.youtube.com/watch?v=WQeoO7MI0Bs)
- [Autonomous Lane Keeping Car Using Raspberry Pi and OpenCV (Instructables)](https://www.instructables.com/Autonomous-Lane-Keeping-Car-Using-Raspberry-Pi-and/)
- [YouTube video](https://www.youtube.com/watch?v=-PCuDnpgiew)

---

## Step 1: Assembly of the Car

Make sure you buy all the materials for the car. Your first step is to make sure the wheels are responding and you are able to make turns. You can refer to the sites above for guidance.

Once the car is able to move left, right, and go straight, the next step is to control the car via a joystick.

When you assemble the car, make sure your camera works with the model of Raspberry Pi you are using. I had to download some extra drivers to make it work.

## Step 2: Collect the Data

I used various traffic signs. I 3D printed small stands and pasted miniature traffic signs like "STOP", "SLOW", etc.

You have to use the joystick to drive the car. I made a temporary track, which you can check out in the embedded video on the [project website](https://www.aasharpam.com/project-autonomous-vehicle.html).

## Step 3: Training

Once you have collected all the data, the next step is to train the model to recognize road signs and navigate the track based on its curvature.

1. Begin by using `DataCollection.py` alongside a joystick to record steering inputs and their corresponding images as you navigate the track. Aim to collect **at least 2,000 images**.
2. Keep in mind that the model's autonomous steering performance depends heavily on the smoothness and accuracy of your driving during data collection. If you are not satisfied with how the model car navigates the track, it is best to repeat the data collection process, as any steering inconsistencies in the training data may carry over into its autonomous performance.
3. Next, use the same program to collect **at least 2,000 images of the stop sign** while approaching it from different angles and positions.
4. Once all the data has been collected, organize it into the appropriate folders and directory structure before running `Training.py`.

This script will train a model capable of autonomously navigating the track while detecting stop signs and bringing the car to a halt when one is identified.

## Step 4: Implementation

Once you have developed a satisfactory model that performs reliably, the next step is to deploy it on your Raspberry Pi.

1. Set up a Python environment and ensure that `WebcamModule.py` and `MotorModule.py` are imported into `runMain.py`.
2. Make sure TensorFlow, OpenCV (`cv2`), and NumPy are installed, as they are required to run the trained model.
3. Transfer your trained model to the Raspberry Pi and specify its correct file path on **line 13** of `runMain.py`.
4. Once everything is configured, run the program to test the car's autonomous performance on the track.

> **Tip:** If the car struggles to stay within the track boundaries or fails to recognize stop signs in time, try reducing the `maxThrottle` value before considering retraining the model. A lower throttle setting reduces the car's speed, giving it more time to respond to changes in the track and detected stop signs, potentially improving its overall control and reliability.

---

## Development Environment

I used PyCharm on Windows 11 as the IDE to write code and then moved it to the Raspberry Pi. You can also use the Python editor on the Raspberry Pi, but it will be slow compared to a Windows machine.

## Code Organization

I have the code for the steps above organized by folder:

- `DataCollection`
- `Training`
- `Production`
