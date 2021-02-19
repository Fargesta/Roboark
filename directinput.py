from directkeys import PressKey, ReleaseKey, E, R
from time import sleep


def press_e():
    PressKey(E)
    sleep(0.2)
    ReleaseKey(E)

def press_r():
    PressKey(R)
    sleep(0.2)
    ReleaseKey(R)