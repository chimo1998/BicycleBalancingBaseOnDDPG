from motor import Motor
import time
import RPi.GPIO as GPIO

btn = 4
rear = Motor(Motor.rear)
front = Motor(Motor.front)
GPIO.setmode(GPIO.BCM)
GPIO.setup(btn, GPIO.IN)

a = 1

while True:
    start = False
    for i in range(10):                                       
        if(GPIO.input(btn)==1):
            start = True
            time.sleep(0.1)
            break
    if(start):
        front.set_speed(a*2)
        rear.set_speed((1-a)*2)
        time.sleep(0.2)
        front.stop()
        rear.stop()
        time.sleep(0.1)
        a = 1-a
