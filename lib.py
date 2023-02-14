
import math
import cv2

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random

def generate_unique_colors(W,H):
    colors = []
    for i in range(int((W*H)*2)):
        r = random.randint(20,245)
        g = random.randint(20,245)
        b = random.randint(20,245)
        colors.append([r,g,b])
    return  list(set(tuple(sorted(sub)) for sub in colors))


def superpixel(img,b_slider, c_slider, misc_slider):
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    height,width,channels = converted_img.shape
    num_iterations = 6
    prior = 2
    double_step = False
    num_superpixels = int(b_slider.val)
    num_levels = int(c_slider.val)
    num_histogram_bins = int(misc_slider.val)


    # seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
    seeds = cv2.ximgproc.createSuperpixelSLIC(converted_img )
    color_img = np.zeros((height,width,3), np.uint8)
    color_img[:] = (0, 0, 255)
    seeds.iterate(num_iterations)

    # retrieve the segmentation result
    labels = seeds.getLabels()


    # labels output: use the last x bits to determine the color
    num_label_bits = 2
    labels &= (1<<num_label_bits)-1
    labels *= 1<<(16-num_label_bits)

    mask = seeds.getLabelContourMask(False)

    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    return  cv2.add(result_bg, result_fg)



def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def hough_lines(img, min, max, threshold, rho, minLineLenght, maxLineGap):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dst = cv2.Canny(img, min, max, None, 3)

    dst = sobel(img, min)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    linesP = cv2.HoughLinesP(dst, rho, np.pi / 180, int(threshold), None, minLineLenght, maxLineGap)

    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    
    # cv2.imshow("Source", img)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("edges", dst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    


def binaryThreshold(mask, thresh):
    # print("calculating thres")
    ret, thresh = cv2.threshold(mask,thresh,255,cv2.THRESH_BINARY)
    return thresh

def sobel(input, thresh, depth = cv2.CV_64F):
    # print("calculating sobel")
    if depth == cv2.CV_64F:
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(input, ddepth=depth, dx=1,dy=0, ksize=3, scale=1)
    y = cv2.Sobel(input, ddepth=depth, dx=0,dy=1, ksize=3, scale=1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return binaryThreshold(edge, thresh)


def is_black(rgb_array):
    return not np.any(rgb_array)

def dilate(edges, size):
    if int(size) <= 0:
        return edges;

    # dilatation_size = int(size)
    # dilation_shape = cv2.MORPH_ELLIPSE

    dilate_shape = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilate_shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    # kernel = cv2.getStructuringElement(dilate_shape, (int(size),int(size)))

    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    # return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return cv2.dilate(edges, element)

floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)

def fill_sobel_segments(img, sobel_thresh, min_filled_pixels_per_segment, dilate_size, skip_step, all_possible_colors):
    edges = sobel(img, sobel_thresh)

    H = img.shape[0]
    W = img.shape[1]

    segments_mask = np.zeros((H,W,3), np.uint8)

    for y in range(0,H, skip_step):
        for x in range(0,W, skip_step):
            # print(f'processing pixel: ({x},{y})')
            # print(f'all_segments_mask[y,x]: {all_segments_mask[y,x]}')
            if edges[y, x] != 255 and is_black(segments_mask[y,x]):
                mask = np.zeros((H+2,W+2), np.uint8)
                cv2.floodFill(edges, mask, seedPoint=(x,y), newVal=(255,0,0), loDiff=(1,)*3, upDiff=(1,)*3, flags=floodflags)
                mask = mask[1:1+H, 1:1+W]
                filled_pixels_count = cv2.countNonZero(mask)  
                if filled_pixels_count < min_filled_pixels_per_segment:
                    continue

                mask = dilate(mask, int(dilate_size))
                mask_inv = cv2.bitwise_not(mask)

                solid_color_image = np.zeros((H,W,3), np.uint8)
                solid_color_image[:,0:W] = all_possible_colors.pop()

                new_colored_segment = cv2.bitwise_or(solid_color_image, solid_color_image, mask=mask)
                segments_mask = add_non_black_to_image(bg_image=segments_mask, fg_image=new_colored_segment, mask=mask_inv)

    return segments_mask, all_possible_colors



def add_non_black_to_image(bg_image, fg_image, mask = []):
    if not np.any(mask):
        mask = cv2.inRange(fg_image, np.array([0,0,0]), np.array([0,0,0]))

    # remove foreground from background
    bg_image = cv2.bitwise_and(bg_image, bg_image, mask = mask)
    return cv2.add(bg_image, fg_image)


def clamp_to_byte(int):
    if int > 255:
        return 255
    elif int < 0:
        return 0
    else:
        return int


def manually_propagate(img, colored_segments):
    H = img.shape[0]
    W = img.shape[1]

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    propagated_image = np.zeros((H,W,3), np.uint8)
    for y in range(H):
        for x in range(W):
            segment_color = colored_segments[y,x]
            avg_grey = 127
            grey_to_add = img_grey[y,x] - avg_grey
            propagated_image[y,x] = [clamp_to_byte(segment_color[0] + grey_to_add), clamp_to_byte(segment_color[1] + grey_to_add), clamp_to_byte(segment_color[2] + grey_to_add)]
    return propagated_image

def propagate_image(img, colored_segments, added_opacity = 0):
    img_v,_,_ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    colored_segments_v,u,y = cv2.split(cv2.cvtColor(colored_segments, cv2.COLOR_BGR2YUV))

    v = colored_segments_v + (img_v.astype(int) - 127)
    v = np.clip(v, 0, 255)
    v = v.astype(u.dtype)


    propagated_image = cv2.merge([v, u, y])
    propagated_image = cv2.cvtColor(propagated_image, cv2.COLOR_YUV2BGR)

    cv2.addWeighted(propagated_image, 1 - (added_opacity/255), colored_segments, 0 + (added_opacity/255), 0.0, propagated_image);

    return propagated_image