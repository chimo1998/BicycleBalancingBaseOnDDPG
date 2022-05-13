from motor import Motor
import time
import RPi.GPIO as GPIO
from gyro import Gyro
import math
import numpy as np

import tensorflow as tf

import env
import train_env

import cv2

btn = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(btn, GPIO.IN)

rear = Motor(Motor.rear)
front = Motor(Motor.front)
rear.stop()
front.stop()
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

motor_speed_front = 0
motor_speed_rear = 0

t = time.time()
o = 3

model_name = "actor_model-480"

training = True

while True:
    for i in range(10):
        if GPIO.input(btn) == 1:
            print('='*30)
            print('btn pressed')
            print('='*30)
            if start:
                rear.stop()
                front.stop()
                o = 5
            else:
                motor_speed_front = 0
                motor_speed_rear = 0
                if training:
                    train_env.init()
                env.init(model_name)
                o = 5
            start = not start
            time.sleep(0.5)
            t = time.time()

    try:
        gx, gy = gyro()
    except KeyboardInterrupt:
        break
    except:
        continue

    dt = time.time() - t
    x = kalman.correct(np.array([[gx], [gy]], np.float32))
    y = kalman.predict()
    x = x[0][0]
    # print(dt)
    dx = (x-last_x) / dt
    ddx = (dx - last_dx)
    last_x = x
    last_dx = dx
    t = time.time()

    if not start or o > 0:
        if start: o-=1
        time.sleep(0.018)
        continue

    a = env.predict((x, dx, ddx, motor_speed_front, motor_speed_rear))
    if(not a):
        rear.stop()
        front.stop()
        start = False
        continue
    a/=1.8
    print(x, dx, ddx, motor_speed_front, motor_speed_rear, dt, a)
    if(a > 0):
        rear.set_speed(a)
        front.stop()
        if(a > motor_speed_rear): motor_speed_rear = a
        else: motor_speed_rear -= 0.4*dt
        motor_speed_front -= 0.4*dt
    else:
        a = -a
        rear.stop()
        front.set_speed(-a)
        if(a > motor_speed_front): motor_speed_front = a
        else: motor_speed_front -= 0.4*dt
        motor_speed_rear -= 0.4*dt
    if motor_speed_front<0: motor_speed_front=0
    if motor_speed_rear<0: motor_speed_rear=0
    time.sleep(0.015)
