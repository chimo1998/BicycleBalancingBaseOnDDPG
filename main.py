from motor import Motor
import time
import RPi.GPIO as gpio
from gyro import Gyro
import math

gpio.setwarnings(False)
btn = 4
gpio.setmode(gpio.BCM)
gpio.setup(btn, gpio.IN)

rear = Motor(Motor.rear)
front = Motor(Motor.front)

gyro = Gyro()

start = False

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
        x,y = gyro()
        print(x)
    except KeyboardInterrupt:
        break
    except:
        print('error')
        continue
    if(start):
        if(x > 0):
            x = math.log(1+x)
            rear.set_speed(x)
            front.stop()
        else:
            x = math.log(1-x)
            rear.stop()
            front.set_speed((-x)/6.0)
    time.sleep(0.03)
