import time
import RPi.GPIO as GPIO
from motor import Motor

rear = Motor(Motor.rear)
front = Motor(Motor.front)
btn = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(btn, GPIO.IN)

current_state = 7
sleep_time = [[0.1, 0.1], [0.10, 0.15], [0.15, 0.10], [0.15, 0.15]]

while True:
    start = False
    for i in range(10):
        if GPIO.input(btn) == 1:
            start = True
            time.sleep(0.1)
            current_state = (current_state + 1) % (len(sleep_time)*2)
            print(current_state)
            break
        time.sleep(0.1)
    if (not start):
        continue
    motor = None
    if current_state%2 == 0:
        print('rear')
        motor = rear
    else:
        print('front')
        motor = front

    for i in range(5):
        st = int(current_state / len(sleep_time))
        print(sleep_time[st])
        motor.set_speed(1)
        time.sleep(sleep_time[st][0])
        motor.stop()
        time.sleep(sleep_time[st][1])
    motor.stop()
