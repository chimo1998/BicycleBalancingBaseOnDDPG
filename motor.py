import time
import RPi.GPIO as gpio

class Motor:
    front = 0
    rear = 1

    freq = 450.0
    front_max = 1950.0
    front_min = 1075.0
    rear_max = 1950.0
    rear_min = 1075.0

    gpio_front = 17
    gpio_rear = 27

    def __init__(self, fr):
        self.start = False
        self.fr = fr
        gpio.setmode(gpio.BCM)
        print('===================================')
        if fr == __class__.front:
            print('front motor')
            gpio.setup(__class__.gpio_front, gpio.OUT)
            pwm = gpio.PWM(__class__.gpio_front, __class__.freq)
            self.pwm = pwm
            self.max = __class__.front_max
            self.min = __class__.front_min
        else:
            print('rear motor')
            gpio.setup(__class__.gpio_rear, gpio.OUT)
            pwm = gpio.PWM(__class__.gpio_rear, __class__.freq)
            self.pwm = pwm
            self.max = __class__.rear_max
            self.min = __class__.rear_min
        self.startup()

    def startup(self):
        print('start up')
        self.pwm.start(66)
        time.sleep(2)
        for i in range(200,400):
            self.pwm.ChangeDutyCycle(100*(i/(1000000/__class__.freq)))
            time.sleep(0.02)
        self.stop()
        print('start up done')
        print('==================================')


    def stop(self):
        self.pwm.ChangeDutyCycle(self.get_duty(0.05))
        #self.start = False

    def get_duty(self, v):
        v = min(2.5, v)
        width = self.min + (self.max-self.min)*(v/5)
        return 100*(width/(1000000/__class__.freq))

    def set_speed(self, v):
        speed = self.get_duty(v)
        if not self.start:
            self.pwm.start(speed)
            self.start = True
            return
        self.pwm.ChangeDutyCycle(speed)
