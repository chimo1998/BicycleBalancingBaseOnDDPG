import RPi.GPIO as gpio
import time

freq = 450.0
front_max_width = 1930
front_min_width = 1520

rear_max_width = 1930
rear_min_width = 1060
def getDuty(width):
    return 100*(width/(1000000/freq))

gpio_motor_front = 17
gpio_motor_rear = 18
gpio_btn = 4

gpio.setmode(gpio.BCM)
gpio.setup(gpio_motor_front, gpio.OUT)
gpio.setup(gpio_motor_rear, gpio.OUT)
gpio.setup(gpio_btn, gpio.IN)

pwm_front = gpio.PWM(gpio_motor_front, freq)
pwm_rear = gpio.PWM(gpio_motor_rear, freq)

start = False

while True:
    for i in range(10):
        if gpio.input(gpio_btn)==1:
            print('btn clicked')
            print(start)
            if start:
                start = False
                pwm_front.stop()
                pwm_rear.stop()
            else:
                start = True
                pwm_front.start(getDuty(1600))
                pwm_rear.start(getDuty(1120))
            time.sleep(0.1)
            break
    time.sleep(0.1)



#pwm = gpio.PWM(go, freq)
#pwm.start(getDuty(1480))
#time.sleep(2)
#for i in range(200,400):
#    pwm.ChangeDutyCycle(getDuty(i))
#    time.sleep(0.02)
#di = -1
#
#for i in range(1600, 1200, -10):
#    pwm.ChangeDutyCycle(getDuty(i))
#    time.sleep(1)
#    print(i)
#pwm.stop()
