from motor import Motor
import RPi.GPIO as GPIO
import time

mot = Motor(Motor.front)

pwm_pin = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(pwm_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
counter = 0

def my_callback(channel):
    global counter
    if GPIO.event_detected(pwm_pin):
        counter += 1

GPIO.add_event_detect(pwm_pin, GPIO.RISING, callback=my_callback)

t = time.time()
while True:
    dt = time.time() - t
    print(dt, counter/28/dt)
    counter = 0
    t = time.time()
    time.sleep(0.1)

