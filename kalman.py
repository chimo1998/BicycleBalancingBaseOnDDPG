import time
import cv2
import numpy as np
import RPi.GPIO as gpio

from gyro import Gyro
from motor import Motor

kalman = cv2.KalmanFilter(2,2)

kalman.measurementMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1e-3
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.01

kalman.statePre = np.array([[6],[6]], np.float32)

gyro = Gyro()

front = Motor(Motor.front)
rear = Motor(Motor.rear)

gpio_btn = 4
gpio.setmode(gpio.BCM)
gpio.setup(gpio_btn, gpio.IN)

start = True

last_x = 0.0
last_v = 0.0
last_a = 0.0

current_x = 0.0
current_v = 0.0
current_a = 0.0

sleep = 0.04

while True:
    for i in range(10):
        if gpio.input(gpio_btn):
            if start:
                front.stop()
                rear.stop()
            else:
                front.set_speed(0.4)
            start = not start
            time.sleep(0.1)
        try:
            gx,gy = gyro()
        except KeyboardInterrupt:
            break
        except:
            continue

        mes = np.array([[gx],[gy]], np.float32)
        x = kalman.correct(mes)
        y = kalman.predict()
        print(x)
        last_x = current_x
        last_v = current_v
        last_a = current_a
        current_x = x[0][0]
        current_v = (current_x - last_x) / sleep
        current_a = (current_v - last_v) / sleep
        print(current_x, current_v, current_a)










