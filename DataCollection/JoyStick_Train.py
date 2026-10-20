from MotorModule import Motor
from time import sleep
import JoyStick as js
import KeyBoard as kp

motor = Motor(2, 3, 4, 17, 22, 27)
runCamera = False
movement = "Joystick"


kp.init()
if runCamera: piCam(x = 200, y = 200)

def main():

    if movement == "Joystick":
        jsVal = js.getJS()
        motor.move(-(jsVal['axis2']), -(jsVal['axis1']), 0.1)

        sleep(0.05)
    else:
        if kp.getKey("UP"):
            motor.move(0.6, 0, 0.1)
        elif kp.getKey("DOWN"):
            motor.move(-0.6, 0, 0.1)
        elif kp.getKey("LEFT"):
            motor.move(0.5, 0.3, 0.1)
        elif kp.getKey("RIGHT"):
            motor.move(0.5, -0.3, 0.1)
        else:
            motor.stop(0.1)



if __name__ == '__main__':
    while True:
        main()