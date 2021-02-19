import cv2 as cv
import numpy as np
from glob import glob
from PIL import Image
from numpy.core.fromnumeric import mean

min_colors = {
    "min_p1": 1000,
    "min_p2": 1000,
    "min_p3": 1000
}

max_colors = {
    "max_p1": 0,
    "max_p2": 0,
    "max_p3": 0
}

save_path = glob("D:/TrianingImages/LostArk/debug_screenshots/")
inc_number = 4

def match_template_ccoeff(img, template):
    res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_val


def create_rod_pixels(img):
    pixels = []
    for x in range(68, 99):
        pixels.append(img[x, 1466])

    np_array = np.array(pixels)
    mean = np.mean(np_array, axis=0).astype(int)
    return mean

def create_energy_pixels(img):
    pixels = [
        img[919, 901],
        img[919, 902],
        img[920, 902],
        img[921, 902]
    ]

    np_array = np.array(pixels)
    mean = np.mean(np_array, axis=0).astype(int)
    return mean


def print_mean_pixels(mean):
    if mean[0] < min_colors["min_p1"]:
        min_colors["min_p1"] =  mean[0]
        print(min_colors)
    if mean[1] < min_colors["min_p2"]:
        min_colors["min_p2"] =  mean[1]
        print(min_colors)
    if mean[2] < min_colors["min_p3"]:
        min_colors["min_p3"] =  mean[2]
        print(min_colors)

    if mean[0] > max_colors["max_p1"]:
        max_colors["max_p1"] =  mean[0]
        print(max_colors)
    if mean[1] > max_colors["max_p2"]:
        max_colors["max_p2"] =  mean[1]
        print(max_colors)
    if mean[2] > max_colors["max_p3"]:
        max_colors["max_p3"] =  mean[2]
        print(max_colors)


def is_rod_broken(img):
    mean = create_rod_pixels(img)
    if 55 <= mean[0] <= 64 and 62 <= mean[1] <= 68 and 132 <= mean[2] <= 139:
        return True

    return False


def has_no_energy(img):
    mean = create_energy_pixels(img)
    if mean[0] <= 5 and mean[1] <=5 and mean[2] <= 6:
        return True

    return False


def save_image(img):
    global inc_number
    full_path = f'{save_path[0]}screenshot_{str(inc_number)}.jpg'
    cv.imwrite(full_path, img)
    print(full_path)
    inc_number += 1
