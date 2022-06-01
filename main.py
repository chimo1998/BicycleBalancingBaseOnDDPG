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
led = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(btn, GPIO.IN)
GPIO.setup(led, GPIO.OUT)
GPIO.output(led, GPIO.LOW)

rear = Motor(Motor.rear)
front = Motor(Motor.front)
rear.stop()
front.stop()
gyro = Gyro()

kalman = cv2.KalmanFilter(2,2)

kalman.measurementMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1e-3
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.001
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

actor_model = "actor_model-480"
critic_model = "critic_model-480"
training = True
base_on = False
keep = True
predict = env.predict

print("="*30)
print('Training setting:')
print('Training: ', training)
print('Base on: ', base_on)
print('Keep: ', keep)
print('='*30)
if training:
    train_env.init(actor_model, critic_model, base_on, keep)
    predict = train_env.predict
else:
    env.init(actor_model)
    predict = env.predict

print('='*30)
print('Initial done')

while True:
    if not start:
        rear.stop()
        front.stop()
    for i in range(10):
        if (not start):
            GPIO.output(led, GPIO.HIGH)
        if GPIO.input(btn) == 1:
            print('='*30)
            print('btn pressed')
            print('='*30)
            if start:
                rear.stop()
                front.stop()
                o = 3
                if training:
                    train_env.save()
            else:
                motor_speed_front = 0
                motor_speed_rear = 0
                if training:
                    train_env.episode_start()

                o = 3
            start = not start
            time.sleep(0.25)
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
        time.sleep(0.1)
        continue

    a = predict((x, dx, ddx, motor_speed_front, motor_speed_rear))
    if(a == None):
        print('stop')
        for stopt in range(5):
            rear.stop()
            front.stop()
            time.sleep(0.1)
        if training:
            train_env.save()
        start = False
        continue
    GPIO.output(led, GPIO.LOW)
    #a/=1.8
    print(x, dx, ddx, motor_speed_front, motor_speed_rear, dt, a)
    if(a > 0):
        motor_speed_front = front.stop()
        if(a > motor_speed_rear or a > motor_speed_rear*0.3):
            motor_speed_rear = a
            rear.set_speed(a)
        else:
            motor_speed_rear = rear.stop()
    else:
        a = -a
        motor_speed_rear = rear.stop()
        if(a > motor_speed_front or a > motor_speed_front*0.3):
            motor_speed_front = a
            front.set_speed(a)
        else:
            motor_speed_front = front.stop()
#         a = -a
#         rear.stop()
#         front.set_speed(-a)
#         if(a > motor_speed_front): motor_speed_front = a
#         else: motor_speed_front -= 0.4*dt
#         motor_speed_rear -= 0.4*dt
    if motor_speed_front<0: motor_speed_front=0
    if motor_speed_rear<0: motor_speed_rear=0
    time.sleep(0.12)
