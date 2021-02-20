from detectionengine import DetectionEngine
import cv2 as cv
from windowcapture import WindowCapture
from time import sleep
import guimonitor
import time
import directinput
import numpy as np
import random


YOLO_CONFIG_PATH = 'models/custom-yolov4-detector.cfg'
YOLO_WEIGHTS_PATH = 'models/yolov4_best_416.weights'
WINDOW_NAME = 'Moonlight' #'LOST ARK (64-bit) v.2.3.1.1' #'Moonlight'

DEBUG = False
#Detection config
confidence_threshold = 0.45
overlap_threshold = 0.3
fish_active_threshold = 0.9
fish_ready_threshold = 0.98 

for i in range(3, -1, -1):
    print('Will start in: ' + str(i))
    sleep(1)

net = cv.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

with open('models/classes.names', 'r', encoding='utf-8') as target:
    labels = target.read().strip().split('\n')

fish_active_template = cv.imread('images/fish_skill_active_pattern.jpg')
fish_ready_template = cv.imread('images/fish_skill_ready_pattern.jpg')
rod_template = cv.imread('images/broken_rod.png', 0)

wincap = WindowCapture(WINDOW_NAME)
wincap.focus_window(wincap.current_window)
wincap.start()

detect = DetectionEngine(net, labels)

#init fishing flags
fed_time = 0
fishing_in_progress = False
throw_time = 0
can_catch = True #Flag for missing catch. For first throw must be True
miss_count = 0
skip_percent = 3 # skip catch if value < fail_percent

def debug():
    detectResult = detect.detect_with_boxes(cropped_screenshot, confidence_threshold, overlap_threshold)
    # cv.imshow("Predictions", detectResult)
    # cv.imshow("skills", skillbox)
    print(fishing_skill_active)
    print(skill_ready)
    pixels = guimonitor.create_energy_pixels(full_size_screenshot)
    guimonitor.print_mean_pixels(pixels)
    #guimonitor.save_image(full_size_screenshot)
    #cv.waitKey(0)


try:
    while True:
        full_size_screenshot = wincap.full_size_screenshot
        cropped_screenshot = wincap.screenshot
        skillbox = wincap.skillBox
        if full_size_screenshot is not None:
            fishing_skill_active = guimonitor.match_template_ccoeff(skillbox, fish_active_template)
            fishing_in_progress = fishing_skill_active > fish_active_threshold

            if DEBUG:
                debug()
            else:
                if not fishing_in_progress:
                    broken_rod = guimonitor.is_rod_broken(full_size_screenshot)
                    skill_ready = guimonitor.match_template_ccoeff(skillbox, fish_ready_template)
                    ready_to_fish = skill_ready > fish_ready_threshold
                    if not broken_rod and ready_to_fish:
                        time_now = time.time()
                        feed_time_passed = time_now - fed_time
                        print ("Last feed: " + time.strftime("%Mm %Ss", time.gmtime(feed_time_passed)) + " ago")
                        if feed_time_passed > 915:
                            print('Feed disabled')
                            #directinput.press_r()
                            #sleep(3)
                            fed_time = time_now

                        if can_catch:
                            print('Throw')
                            directinput.press_e()
                            can_catch = False
                            sleep(3)
                        else:
                            miss_count += 1
                            print("Missed! Miss count: " + str(miss_count)) #possible bot detection
                            
                            if miss_count > 3:
                                raise Exception("Too much misses")

                            can_catch = True
                            seed = random.Random.randrange(2000) / 1000
                            sleep_time = 5.5 + seed
                            sleep(sleep_time)
                            print("Sleep time: " + str(sleep_time))
                    elif broken_rod:
                        print("Fishing rod broken...")
                        can_catch = True
                        sleep(5)
                    else:
                        raise Exception("Not enough energy. Fishing will be stopped.")
                else:
                    detect_result = detect.detect_with_confidence(cropped_screenshot, confidence_threshold)
                    if detect_result:
                        seed = random.Random.randrange(2000) / 1000

                        if seed < skip_percent * 0.02: #Reducing number of success to avoid bot detection
                            print("Skip")
                            if miss_count >= 0: #Skip will increase miss count thus miss_count must be reduced
                                miss_count -= 1
                        else:
                            print('Catch!')
                            directinput.press_e()
                            can_catch = True
                            sleep_time = 5.5 + seed
                            sleep(sleep_time)
                            print("Sleep time: " + str(sleep_time))

except KeyboardInterrupt:
    print ("User interrupt")
except Exception as ex:
    print("Warning: " + str(ex))
finally:
    wincap.stop()
    if DEBUG:
        cv.destroyAllWindows()
    print('Stopped...')