from motor import Motor
import time
import RPi.GPIO as GPIO
from gyro import Gyro
import math
import numpy as np
import serial
import tensorflow as tf
from decimal import Decimal

import env
import train_env

import cv2

from hipnuc_module import *

btn = 4
led = 21
led_angle = 6
stop_led = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(btn, GPIO.IN)
GPIO.setup(led, GPIO.OUT)
GPIO.output(led, GPIO.LOW)
GPIO.setup(stop_led, GPIO.OUT)
GPIO.output(stop_led, GPIO.LOW)

GPIO.setup(led_angle, GPIO.OUT)
GPIO.output(led_angle,GPIO.LOW)

#ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
#ser.reset_input_buffer()



da_max = 0.17
rear = Motor(Motor.rear,da_max)
front = Motor(Motor.front,da_max)
rear.stop()
front.stop()
#gyro = Gyro()

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

x = 0
last_x = 0
dx = 0
last_dx = 0
ddx = 0

motor_speed_front = 0
motor_speed_rear = 0

t = time.time()
o = 3
dt_threshold = 0.1


model_num = 470
folder = "h5/"
actor_model = folder + "actor_model-%d" % model_num
critic_model = folder + "critic_model-%d" % model_num

m_IMU = hipnuc_module('./config.json')
purpose = 0
training = purpose != 3 # 從頭訓練
base_on = purpose == 1 # 從上面設定的model開始訓練
keep = purpose == 2 # 從train_env設定的model開始訓練，會讀Buffer
start_run = purpose == 3 # 從上面設定的model開始跑
predict = env.predict

if purpose == 1 or purpose == 3:
    print('='*30)
    print(actor_model)
    print(critic_model)

print("="*30)
print('Training setting:')
print('Training: ', training)
print('Base on: ', base_on)
print('Keep: ', keep)
print('='*30)
if training:
    train_env.init(actor_model, critic_model, base_on, keep, start_run)
    predict = train_env.predict
elif start_run:
    train_env.init(actor_model, critic_model, base_on, keep, start_run)
    predict = train_env.run_predict

print('='*30)
print('Initial done')
ct = 0
# ttt = 0
while True:
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
                # gyro.reset()
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
            time.sleep(0.5)
            t = time.time()

    try:
        data = m_IMU.get_module_data(10)
        euler_str=str(data['euler'])
        x2=euler_str[10:14].strip(",")
        x=float(x2.strip())
        if((int(x)<=0.5) & (int(x)>-0.5)):
            GPIO.output(led_angle, GPIO.HIGH)
        else:
            GPIO.output(led_angle, GPIO.LOW)
#        gx, gy, gz = gyro()
#       if math.isnan(gx):
#           print('nan')
#           time.sleep(0.1)
#           continue
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
        continue

    dt = time.time() - t
 
    # ttt += 1
    # x = gx
    if(dt < dt_threshold):
        continue
    # x = gx
    # print(dt)
    # print(ttt)
    # ttt = 0

    dx = ((x-last_x) / dt) / 3
    ddx = (dx - last_dx) #/ dt
    last_x = x
    last_dx = dx
    t = time.time()
    #print(x, dx, ddx)
    if not start or o > 0:
        if start: o-=1
        rear.stop()
        front.stop()
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

    if (a<0):
        a = -a
        # a *= 1.25
        if (a <= motor_speed_front*da_max):
            motor_speed_front *= da_max
        else:
            motor_speed_front = a
        front.set_speed(motor_speed_front)
        motor_speed_rear = rear.stop(da_max)
    else:
        if (a <= motor_speed_rear*da_max):
            motor_speed_rear*=da_max
        else:
            motor_speed_rear = a
        rear.set_speed(motor_speed_rear)
        motor_speed_front = front.stop(da_max)
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
    t = time.time()
