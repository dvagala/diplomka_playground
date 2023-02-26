
import math
import cv2
import os
import sys

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
import itertools
import cv2

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random
import time
import itertools


def add_non_black_to_image(bg_image, fg_image, mask=[]):
    if not np.any(mask):
        mask = cv2.inRange(fg_image, np.array([0, 0, 0]), np.array([0, 0, 0]))

    # remove foreground from background
    bg_image = cv2.bitwise_and(bg_image, bg_image, mask=mask)
    return cv2.add(bg_image, fg_image)


def get_colors_from_bigger_segments(img):
    H = img.shape[0]
    W = img.shape[1]

    kernel_size = 20

    segments_colors = []

    for x_kernel in range(int(W / kernel_size)):
        for y_kernel in range(int(H / kernel_size)):
            first_color_in_kernel = img[y_kernel *
                                        kernel_size, x_kernel * kernel_size]
            there_are_multiple_colors_in_kernel = False

            if list(first_color_in_kernel) in segments_colors:
                continue

            for relative_x_in_kernel in range(kernel_size):
                if there_are_multiple_colors_in_kernel:
                    break

                for relative_y_in_kernel in range(kernel_size):
                    absolute_x = x_kernel * kernel_size + relative_x_in_kernel
                    absolute_y = y_kernel * kernel_size + relative_y_in_kernel

                    current_color = img[absolute_y, absolute_x]
                    if not (first_color_in_kernel == current_color).all():
                        there_are_multiple_colors_in_kernel = True
                        break

            if not there_are_multiple_colors_in_kernel:
                segments_colors.append(list(first_color_in_kernel))

    return list(set(map(tuple, segments_colors)))


def keep_only_these_colors_in_image(img, colors_to_keep):
    H = img.shape[0]
    W = img.shape[1]

    final_image = np.zeros((H, W, 3), np.uint8)

    for color in colors_to_keep:
        np_color = np.asarray(color)

        mask = cv2.inRange(img, np_color, np_color)
        colored_segment = cv2.bitwise_and(img, img, mask=mask)

        kernel = np.ones((11, 11), np.uint8)
        colored_segment = cv2.dilate(colored_segment, kernel)

        final_image = add_non_black_to_image(
            bg_image=final_image, fg_image=colored_segment)

    return final_image


def remove_soft_transitions_between_segments(img):
    segments_color = get_colors_from_bigger_segments(img)
    return keep_only_these_colors_in_image(img, segments_color)


def resize(img, target_width, make_square):
    original_width = img.shape[0]
    original_height = img.shape[1]


    if make_square and original_height != original_width:
        if original_width > original_height:
            h = original_height
            w = original_height 
        elif original_height > original_width:
            h = original_width
            w = original_width 

        center = img.shape
        x = center[1]/2 - w/2
        y = center[0]/2 - h/2
        img = img[int(y):int(y+h), int(x):int(x+w)]
        target_height = target_width
    else:
        target_height = int(target_width/(original_height/original_width))

    return cv2.resize(img, (target_width, target_height), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)


def  take_only_non_corrupted_screenshot(all_screenshots_files):
    non_corrupted_screenshots = []

    for file in all_screenshots_files:
        if ".jpg" in file or ".png" in file:
            screenshot_id = file.replace("-segmented", "").split(".")[0]
            if "-segmented" in file and any(f'{screenshot_id}.' in s for s in all_screenshots_files):
                non_corrupted_screenshots.append(file)
            elif "-segmented" not in file and any(f'{screenshot_id}-segmented.' in s for s in all_screenshots_files):
                non_corrupted_screenshots.append(file)
            else:
                print(f'this file is missing his pair: {file}')

    return non_corrupted_screenshots

def show_missing_files():
    for file in os.listdir(dst_anotations_dir):
        full_path = f'{dst_anotations_dir}/{file}'
        os.remove(full_path)
    for file in os.listdir(dst_images_dir):
        full_path = f'{dst_images_dir}/{file}'
        os.remove(full_path)
    files = list(map(lambda x: f'{x.split(".")[0]}-segmented.{x.split(".")[1]}', os.listdir("/Users/dvagala/School/diplomka_playground/making_yolo_dataset/img_to_coco/anotation_masks"))) 
    files2 = os.listdir("/Users/dvagala/School/diplomka_playground/making_yolo_dataset/img_to_coco/coco_dataset")
    take_only_non_corrupted_screenshot(files + files2)

# Note this is optimized for annotation images that has width 11 500px

src_screenshots_dir = sys.argv[1].removesuffix('/')
dst_anotations_dir = sys.argv[2].removesuffix('/')
dst_images_dir = sys.argv[3].removesuffix('/')


all_screenshots = take_only_non_corrupted_screenshot(os.listdir(src_screenshots_dir))

for file in all_screenshots:
    print(f'processing: {file}')

    src_full_path = f'{src_screenshots_dir}/{file}'
    img = cv2.imread(src_full_path)

    if "-segmented" in file:
        dst_full_path = f'{dst_anotations_dir}/{file.replace("-segmented", "")}'
        img = remove_soft_transitions_between_segments(img)
    else:
        dst_full_path = f'{dst_images_dir}/{file}'

    img = resize(img, target_width = 640, make_square = True)

    cv2.imwrite(dst_full_path, img)
