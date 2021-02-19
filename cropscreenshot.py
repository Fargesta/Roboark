import glob
import os
import cv2 as cv

def crop_from_center(img, size):
    x = img.shape[1] / 2 - size / 2
    y = img.shape[0] / 2 - size / 2
    return img[int(y):int(y + size), int(x):int(x + size)]

path_in = glob.glob("D:/TrianingImages/LostArk/images/*.jpg")
path_out = glob.glob("D:/TrianingImages/LostArk/croped_images_416/")
for img in path_in:
    read_img = cv.imread(img)
    file_name = os.path.basename(img)
    crop = crop_from_center(read_img, 416)
    full_path = path_out[0] + 's_416_' + file_name
    cv.imwrite(full_path, crop)
    print(full_path)

