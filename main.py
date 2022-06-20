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
stop_led = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(btn, GPIO.IN)
GPIO.setup(led, GPIO.OUT)
GPIO.output(led, GPIO.LOW)
GPIO.setup(stop_led, GPIO.OUT)
GPIO.output(stop_led, GPIO.LOW)

rear = Motor(Motor.rear)
front = Motor(Motor.front)
rear.stop()
front.stop()
gyro = Gyro()

kalman = cv2.KalmanFilter(2,2)

kalman.measurementMatrix = np.array([[1,0],[0,1]], np.float32)
kalman.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
# ==============================================================================================
# processNoiseCov 預測噪音
# 越大越不穩定且越容易接近系統預測值，單步變化大
# 噪音越小則預測結果與上個計算差不多
kalman.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.001
# ==============================================================================================
# measurementNoiseCov 量測斜方差矩陣
# 方差越小預測結果越接近測量值
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.008
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
keep = False
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
ct = 0
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
                GPIO.output(stop_led, GPIO.HIGH)
                for stopt in range(5):
                    rear.stop()
                    front.stop()
                    time.sleep(0.1)
                o = 3
                if training:
                    train_env.save()
                GPIO.output(stop_led, GPIO.LOW)
            else:
                gyro.reset()
                motor_speed_front = 0
                motor_speed_rear = 0
                front.set_speed(0)
                rear.set_speed(0)
                if training:
                    train_env.episode_start()

                o = 3
            start = not start
            print(start)
            print('='*30)
            time.sleep(0.25)
            t = time.time()

    try:
        z, gx, gy = gyro()
        if math.isnan(gx):
            print('nan')
            time.sleep(0.1)
            continue
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
        continue

    dt = time.time() - t

    x = kalman.correct(np.array([[gx], [gy]], np.float32))
    y = kalman.predict()
    x = x[0][0]
    # x = gx
    if(dt < 0.1):
        continue
    # x = gx
    # print(dt)


    dx = (x-last_x) / dt
    ddx = (dx - last_dx) / dt
    last_x = x
    last_dx = dx
    t = time.time()
    #print(x, dx, ddx)
    if not start or o > 0:
        if start: o-=1
        continue

    a = predict((x, dx, ddx, motor_speed_front, motor_speed_rear))
    if(a==200):#len(a)==0):
        print('stop')
        GPIO.output(stop_led, GPIO.HIGH)
        for stopt in range(2):
            rear.stop()
            front.stop()
            time.sleep(0.1)
        if training:
            train_env.save()
        start = False
        GPIO.output(stop_led, GPIO.LOW)
        continue
    GPIO.output(led, GPIO.LOW)
    #a/=1.8
    print(x, dx, ddx, motor_speed_front, motor_speed_rear, dt, a)

    if (a>0):
        if (a <= motor_speed_front*0.3):
            motor_speed_front *= 0.3
        else:
            motor_speed_front = a
        front.set_speed(motor_speed_front)
        motor_speed_rear = rear.stop()
    else:
        a = -a
        if (a <= motor_speed_rear*0.3):
            motor_speed_rear*=0.3
        else:
            motor_speed_rear = a
        rear.set_speed(motor_speed_rear)
        motor_speed_front = front.stop()
    continue



    if(a[0] <= motor_speed_front*0.3):
        motor_speed_front = motor_speed_front*0.3
    else:
        motor_speed_front = a[0]
    if motor_speed_front<0:
        motor_speed_front=0
    front.set_speed(motor_speed_front)
    
    if(a[1] <= motor_speed_rear*0.3):
        motor_speed_rear = motor_speed_rear*0.3
    else:
        motor_speed_rear = a[1]
    if motor_speed_rear<0:
        motor_speed_rear=0
    rear.set_speed(motor_speed_rear)
