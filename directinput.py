from directkeys import PressKey, ReleaseKey, E, R, F
from time import sleep
import random


def press_e():
    press_time = random.uniform(0.185, 0.298)
    PressKey(E)
    sleep(press_time)
    ReleaseKey(E)

def press_r():
    press_time = random.uniform(0.185, 0.298)
    PressKey(R)
    sleep(press_time)
    ReleaseKey(R)

def press_f():
    press_time = random.uniform(0.185, 0.298)
    PressKey(F)
    sleep(press_time)
    ReleaseKey(F)