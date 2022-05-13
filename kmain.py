from motor import Motor
import time
import RPi.GPIO as gpio
from gyro import Gyro
import math
import numpy as np
import cv2

gpio.setwarnings(False)
btn = 4
gpio.setmode(gpio.BCM)
gpio.setup(btn, gpio.IN)

rear = Motor(Motor.rear)
front = Motor(Motor.front)

gyro = Gyro()

kalman = cv2.KalmanFilter(2,2)

kalman.measurementMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1e-3
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.01

kalman.statePre = np.array([[6],[6]], np.float32)

start = False

last_x = 0
dx = 0
last_dx = 0
ddx = 0

t = time.time()

while True:
    for i in range(10):
        if gpio.input(btn) == 1:
            print('btn pressed')
            if start:
                rear.stop()
                front.stop()
            start = not start
            print(start)
            time.sleep(1)
    try:
        gx,gy = gyro()
    except KeyboardInterrupt:
        break
    except:
        print('error')
        continue

    x = kalman.correct(np.array([[gx],[gy]], np.float32))
    y = kalman.predict()
    x = x[0][0]
    dt = time.time() - t
    dx = (x - last_x) / dt
    ddx = (dx - last_dx) / dt
    last_x = x
    last_dx = dx
    print(x, dx, ddx)

    if(start):
        v = 0.0
        if(x > 20 or x < -20):
            rear.stop()
            front.stop()
            start = False
            continue
        if(x < 0):
            v = math.log(1.5-x)
            rear.set_speed(-x/4)
            front.stop()
        else:
            v = math.log(1.5+x)
            rear.stop()
            front.set_speed(x/4)
        print('v', v)
    time.sleep(0.025)
