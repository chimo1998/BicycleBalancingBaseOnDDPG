import RPi.GPIO as gpio
import time

freq = 450.0
max_width = 1930
min_width = 1060
def getDuty(width):
    return 100*(width/(1000000/freq))

go = 17
sp = 27
gpio.setmode(gpio.BCM)
gpio.setup(go, gpio.OUT)
pwm = gpio.PWM(go, freq)
pwm.start(getDuty(1480))
time.sleep(2)
for i in range(200,400):
    pwm.ChangeDutyCycle(getDuty(i))
    time.sleep(0.02)
di = -1

for i in range(1600, 1200, -10):
    pwm.ChangeDutyCycle(getDuty(i))
    time.sleep(1)
    print(i)
pwm.stop()
