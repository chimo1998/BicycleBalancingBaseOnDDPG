import time
import RPi.GPIO as gpio

class Motor:
    front = 0
    rear = 1

    freq = 450.0

    gpio_front = 27
    gpio_rear = 17

    def __init__(self, fr, da_max):
        self.start = False
        self.da_max = da_max
        self.speed = 0
        self.max_width = 1950
        self.min_width = 1050
        self.fr = fr
        gpio.setmode(gpio.BCM)
        print('===================================')
        if fr == __class__.front:
            print('front motor')
            gpio.setup(__class__.gpio_front, gpio.OUT)
            pwm = gpio.PWM(__class__.gpio_front, __class__.freq)
            self.pwm = pwm
        else:
            print('rear motor')
            gpio.setup(__class__.gpio_rear, gpio.OUT)
            pwm = gpio.PWM(__class__.gpio_rear, __class__.freq)
            self.pwm = pwm
        self.startup()

    def startup(self):
        print('start up')
        self.pwm.start(100*(self.max_width/(1000000/__class__.freq)))
        time.sleep(2)
        self.pwm.ChangeDutyCycle(100*(self.min_width/(1000000/__class__.freq)))
        time.sleep(1)
        self.min_width += 50
        self.stop()
        print('start up done')
        print('==================================')


    def stop(self, da=0.25):
        self.speed = self.speed * self.da_max
        self.pwm.ChangeDutyCycle(self.get_duty(self.speed))
        return self.speed
        #self.start = False

    def get_duty(self, v):
        v = min(4, v)
        width = self.min_width + (self.max_width-self.min_width)*(v/5)
        d = 100*(width/(1000000/__class__.freq))
        # print(d)
        return d

    def set_speed(self, v):
        self.speed = v
        speed = self.get_duty(v)
        if not self.start:
            self.pwm.start(speed)
            self.start = True
            return self.speed
        self.pwm.ChangeDutyCycle(speed)
        return self.speed
