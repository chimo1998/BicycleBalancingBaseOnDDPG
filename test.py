
from motor import Motor
import time
import RPi.GPIO as gpio

gpio.setwarnings(False)

front = Motor(Motor.front)
rear = Motor(Motor.rear)

gpio_btn = 4

gpio.setmode(gpio.BCM)
gpio.setup(gpio_btn, gpio.IN)

state = 0
speed = 0.5

#while True:
#    for i in range(10):
#        if gpio.input(gpio_btn)==1:
#            speed = speed + 0.2
#            print('speed:', speed)
#            rear.set_speed(speed)
#            time.sleep(0.1)
#        time.sleep(0.1)

while True:
    for i in range(10):
        if gpio.input(gpio_btn)==1:
            state = (state+1)%4
            print('state:', state)
            if state==0:
                front.stop()
                rear.stop()
            elif state==1:
                front.set_speed(0.4)
                rear.stop()
            elif state==2:
                front.stop()
                rear.set_speed(0.4)
            elif state==3:
                front.set_speed(0.4)
                rear.set_speed(0.4)
            time.sleep(0.1)
        time.sleep(0.1)

