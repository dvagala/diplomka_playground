
import math
import cv2

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random

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

