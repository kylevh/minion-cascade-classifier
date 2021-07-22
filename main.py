import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from vision import Vision

wincap = WindowCapture()

cascade_minion = cv.CascadeClassifier('cascade/cascade.xml')

vision_minion = Vision(None)

loop_time = time()

while(True):
    screenshot = wincap.get_screenshot()

    rectangles = cascade_minion.detectMultiScale(screenshot)

    detection_image = vision_minion.draw_rectangles(screenshot, rectangles)
    #display images
    cv.namedWindow('Matches', cv.WINDOW_NORMAL)
    cv.imshow('Matches', detection_image)

    #debug loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # Exit function
    key = cv.waitKey(1)
    if key == ord('m'):
        cv.destroyAllWindows() 
        break
    elif key == ord('p'):
        cv.imwrite('positive/{}.jpg'.format(loop_time), screenshot)
    elif key == ord('o'):
        cv.imwrite('negative/{}.jpg'.format(loop_time), screenshot)

print('Finished')
