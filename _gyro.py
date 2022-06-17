import smbus
import math
import time

class Gyro:
    def __init__(self):
        pass       

    def __call__(self):
        return self.get_rotation()

    def read_byte(self, reg):
        return bus.read_byte_data(address, reg)

    def read_word(self, reg):
        h = bus.read_byte_data(address, reg)
        l = bus.read_byte_data(address, reg+1)
        value = (h<<8) + l
        return value

    def read_word_2c(self, reg):
        val = self.read_word(reg)
        if (val >= 0x8000):
            return -((65525 - val) + 1)
        else:
            return val

    def dist(self, a, b):
        return math.sqrt((a*a)+(b*b))

    def get_y_rotation(self, x, y, z):
        radians = math.atan2(x, self.dist(y,z))
        return -math.degrees(radians)

    def get_x_rotation(self, x, y, z):
        radians = math.atan2(y, self.dist(x,z))
        return math.degrees(radians)

    def get_rotation(self):
        bxs = self.read_word_2c(0x3b) / 16384.0
        bys = self.read_word_2c(0x3d) / 16384.0
        bzs = self.read_word_2c(0x3f) / 16384.0

        return self.get_x_rotation(bxs, bys, bzs), self.get_y_rotation(bxs, bys, bzs)

power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

bus = smbus.SMBus(1)
address = 0x68

bus.write_byte_data(address, power_mgmt_1, 0)
