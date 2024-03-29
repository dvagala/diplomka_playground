import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
import numpy as np
import os
import torch


size = (512*2, 512*2)
# size = (512, 512)
# image = cv2.imread('/Users/dvagala/School/LDC/data/lisbon.jpg', cv2.IMREAD_COLOR)

img = cv2.imread('/Users/dvagala/School/LDC/data/lisbon.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('/Users/dvagala/School/LDC/data/facade.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('/Users/dvagala/School/LDC/data/0a304d20-82e6-11ed-bab6-e370802a6055_orig-photo.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('/Users/dvagala/School/LDC/data/0a625870-9b36-11ed-b2b4-6f2be65a6198_orig-photo.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('/Users/dvagala/School/LDC/data/cubes.png', cv2.IMREAD_COLOR)

img = cv2.resize(img, size )
image = cv2.resize(img, size )
image = torch.from_numpy(image.copy())
image = torch.permute(image, (2, 0, 1))
image = image.float()
image = image.unsqueeze(0)

# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/B3_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/B4_16_model_scripted.pt')
model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_BSDS_with_augmentation_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_BSDS+CROPPUI111_using_original_brind_augmentation_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_BSDS+CROPPUI111_using_my_new_augmentation90x_with_hue_augmentation_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_BSDS+CROPPUI111_using_original_brind_augmentation_16_model_scripted.pt')
# model = torch.jit.load('/Users/dvagala/School/LDC/checkpoints/trained_on_CROPPUI111_with_my_augmentation90x_with_hue_16_model_scripted.pt')
model.eval()
out = model(image)
out = out[len(out)-1]

tensor_out = out.squeeze()
tensor_out = torch.sigmoid(tensor_out)
tensor_out = torch.clamp(torch.mul(tensor_out, 255), min= 0, max= 255)





import cv2

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random
import time
import itertools


from lib import *


ldc_edges = cv2.imread("ldc_edges.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.GaussianBlur(img, (3,3), 0)
# img = cv2.imread("photo_small.jpg")
# cv2.imshow('img', img)
(H, W) = img.shape[:2]


all_possible_colors_orig = generate_unique_colors(W, H)
all_possible_colors = all_possible_colors_orig[:]


def hed():
    # print("calculating hed")
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0, size=(W, H), swapRB=False, crop=False)
    net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt", "hed_pretrained_bsds.caffemodel")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    # return hed
    return binaryThreshold(hed)


dilate_shape = cv2.MORPH_ELLIPSE


def canny(input, min, max):
    return cv2.Canny(input, min, max, apertureSize=3, L2gradient=True)


touch_points = []

fig = plt.figure()
fig.subplots_adjust(left=0.55, bottom=0.55)

t = np.arange(0.0, 1.0, 0.001)
canny_min_thresh = 30
canny_max_thresh = 50
sobel_thresh = 15

# cv2.imshow("sobel", sobel(sobel_thresh))
# cv2.imshow("canny", canny(canny_min_thresh, canny_max_thresh))

# sob = sobel(sobel_thresh)
# seed = (int(W/2),int(H/2))


def draw_rectangle(mask, point):
    mask[point[1] - 1, point[0] - 1] = 255
    mask[point[1] - 1, point[0]] = 255
    mask[point[1] - 1, point[0] + 1] = 255

    mask[point[1], point[0] - 1] = 255
    mask[point[1], point[0]] = 255
    mask[point[1], point[0] + 1] = 255

    mask[point[1] + 1, point[0] - 1] = 255
    mask[point[1] + 1, point[0]] = 255
    mask[point[1] + 1, point[0] + 1] = 255


sobel_1_thresh_slider_ax = fig.add_axes([0.25, 0.65, 0.65, 0.03])
sobel_1_thresh_slider = Slider(
    sobel_1_thresh_slider_ax, 'Sobel 1 thresh', 0.0, 257, valinit=130)

sobel_2_thresh_slider_ax = fig.add_axes([0.25, 0.6, 0.65, 0.03])
sobel_2_thresh_slider = Slider(
    sobel_2_thresh_slider_ax, 'Sobel 2 thresh', 0, 255, valinit=127)

sobel_3_thresh_slider_ax = fig.add_axes([0.25, 0.55, 0.65, 0.03])
sobel_3_thresh_slider = Slider(
    sobel_3_thresh_slider_ax, 'Sobel 3 thresh', 0.1, 500, valinit=13)

dilate_1_slider_ax = fig.add_axes([0.25, 0.5, 0.65, 0.03])
dilate_1_slider = Slider(dilate_1_slider_ax, 'Dilate 1', 0, 20, valinit=2)

dilate_2_slider_ax = fig.add_axes([0.25, 0.45, 0.65, 0.03])
dilate_2_slider = Slider(dilate_2_slider_ax, 'Dilate 2', 0, 15, valinit=3)

dilate_3_slider_ax = fig.add_axes([0.25, 0.4, 0.65, 0.03])
dilate_3_slider = Slider(dilate_3_slider_ax, 'Dilate 3', 0, 20, valinit=2)

min_area_1_slider_ax = fig.add_axes([0.25, 0.35, 0.65, 0.03])
min_area_1_slider = Slider(
    min_area_1_slider_ax, 'Min area 1', 0, 30000, valinit=0)

min_area_2_slider_ax = fig.add_axes([0.25, 0.3, 0.65, 0.03])
min_area_2_slider = Slider(
    min_area_2_slider_ax, 'Min area 2', 0, 30000, valinit=(W * H) / 14563)

min_area_3_slider_ax = fig.add_axes([0.25, 0.25, 0.65, 0.03])
min_area_3_slider = Slider(
    min_area_3_slider_ax, 'Min area 3', 0, 30000, valinit=(W * H) / 4104)

skip_step_slider_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
skip_step_slider = Slider(skip_step_slider_ax, 'Skip step', 0, 80, valinit=12)

# Define an action for modifying the line when any slider's value changes


def sliders_on_changed(val):
    # cv2.imshow("sobel", sob)
    # cv2.imshow("canny", canny(canny_min_thresh_slider.val, canny_max_thresh_slider.val))
    render()


all_segments_mask = np.zeros((H, W, 3), np.uint8)
selected_segments = np.zeros((H, W, 3), np.uint8)


def is_within_boundaries(x, y, W, H):
    if (x < 0 or x >= W) or (y < 0 or y >= H):
        return False
    else:
        return True


def dillute_to_neighbour_if_empty(neighbour_x, neighbour_y, current_pixel, col):
    global all_segments_mask
    if is_within_boundaries(neighbour_x, neighbour_y, W, H) and is_black(all_segments_mask[neighbour_y, neighbour_x]):
        all_segments_mask[neighbour_y, neighbour_x] = current_pixel
        # all_segments_mask[neighbour_y, neighbour_x] = (0, 0, 255)
        # all_segments_mask[neighbour_y, neighbour_x] = col
        return True
    else:
        return False


def dillute_segments():
    global H, W, all_segments_mask

    i = 0
    while True:
        black_count = 0
        diluted_count = 0

        if i % 2 == 0:
            print('going from top left')

            def get_neighbours(x, y):
                return [
                    (x - 1, y - 1),
                    (x, y - 1),
                    (x + 1, y - 1),
                    (x - 1, y)
                ]
            x_range = list(range(W))
            y_range = list(range(H))
            color = (0, 0, 255)
        else:
            print('going from bottom right')

            def get_neighbours(x, y):
                return [
                    (x + 1, y),
                    (x + 1, y + 1),
                    (x, y + 1),
                    (x - 1, y + 1)
                ]
            x_range = list(reversed(range(W)))
            y_range = list(reversed(range(H)))
            color = (255, 0, 0)

        for y in y_range:
            for x in x_range:
                current_pixel = all_segments_mask[y, x]
                if is_black(current_pixel):
                    black_count += 1
                else:
                    for neighbour in get_neighbours(x, y):
                        was_diluted = dillute_to_neighbour_if_empty(
                            neighbour_x=neighbour[0], neighbour_y=neighbour[1], current_pixel=current_pixel, col=color)
                        if was_diluted == True:
                            diluted_count += 1

        print(
            f'iteration: {i}, black_count: {black_count}, diluted_count: {diluted_count}')
        if black_count == diluted_count:
            break

        i += 1


def flood_fill_edges_on_touch_points(edges):
    global all_segments_mask
    for touch_point in touch_points:
        if edges[touch_point[1], touch_point[0]] != 255:
            mask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(edges, mask, seedPoint=touch_point, newVal=(
                255, 0, 0), loDiff=(1,) * 3, upDiff=(1,) * 3, flags=floodflags)
            mask = mask[1:1 + H, 1:1 + W]

            solid_color_image = np.zeros((H, W, 3), np.uint8)
            solid_color_image[:,
                              0:W] = all_possible_colors[touch_point[0] + touch_point[1]]
            new_colored_segment = cv2.bitwise_or(
                solid_color_image, solid_color_image, mask=mask)
            all_segments_mask = cv2.bitwise_or(
                new_colored_segment, all_segments_mask)


def flood_fill_segmented_on_touch_points(all_segments):
    selected_segments = np.zeros((H, W, 3), np.uint8)

    for touch_point in touch_points:
        if is_black(selected_segments[touch_point[1], touch_point[0]]):
            mask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(all_segments, mask, seedPoint=touch_point, newVal=(
                255, 0, 0), loDiff=(1,) * 3, upDiff=(1,) * 3, flags=floodflags)
            mask = mask[1:1 + H, 1:1 + W]

            new_colored_segment = cv2.bitwise_and(
                all_segments, all_segments, mask=mask)
            selected_segments = cv2.bitwise_or(
                new_colored_segment, selected_segments)

    return selected_segments


def split(list):
    return list[::3], list[1::3], list[2::3]


def create_all_segments_mask():
    colors1, colors2, colors3 = split(all_possible_colors_orig[:])

    all_segments_mask = np.zeros((H, W, 3), np.uint8)
    all_segments_mask[:, 0:W] = colors1.pop()

    segments_mask, colors = fill_sobel_segments(img, edges=sobel(img, sobel_1_thresh_slider.val), min_filled_pixels_per_segment=int(
        min_area_1_slider.val), dilate_size=int(dilate_1_slider.val), skip_step=int(skip_step_slider.val), all_possible_colors=colors1)
    all_segments_mask = add_non_black_to_image(
        bg_image=all_segments_mask, fg_image=segments_mask)

    segments_mask, colors = fill_sobel_segments(img, edges=sobel(img, sobel_2_thresh_slider.val), min_filled_pixels_per_segment=int(
        min_area_2_slider.val), dilate_size=int(dilate_2_slider.val), skip_step=int(skip_step_slider.val), all_possible_colors=colors2)
    all_segments_mask = add_non_black_to_image(
        bg_image=all_segments_mask, fg_image=segments_mask)

    segments_mask, colors = fill_sobel_segments(img, edges=sobel(img, sobel_3_thresh_slider.val), min_filled_pixels_per_segment=int(
        min_area_3_slider.val), dilate_size=int(dilate_3_slider.val), skip_step=int(skip_step_slider.val), all_possible_colors=colors3)
    all_segments_mask = add_non_black_to_image(
        bg_image=all_segments_mask, fg_image=segments_mask)

    # segments_mask, all_possible_colors =  fill_sobel_segments(img, sobel_thresh=int(sobel_thresh_slider.val), min_filled_pixels_per_segment=c_slider.val, dilate_size=2, skip_step=8, all_possible_colors=all_possible_colors)
    # all_segments_mask = add_non_black_to_image(bg_image=all_segments_mask, fg_image=segments_mask)

    return all_segments_mask


def create_segments_from_edges(edges_binary):
    colors1, colors2, colors3 = split(all_possible_colors_orig[:])

    all_segments_mask = np.zeros((H, W, 3), np.uint8)
    all_segments_mask[:, 0:W] = colors1.pop()

    # _, ldc_edges_binary = cv2.threshold(edges, 161, 255, cv2.THRESH_BINARY_INV)
    segments_mask, colors = fill_sobel_segments(img, min_filled_pixels_per_segment=0, dilate_size=1, skip_step=int(skip_step_slider.val), all_possible_colors=colors1, edges=edges_binary)

    all_segments_mask = add_non_black_to_image(
        bg_image=all_segments_mask, fg_image=segments_mask)

    return all_segments_mask

# all_segments_mask = create_all_segments_mask()
# cv2.imshow("all_segments_mask", all_segments_mask)


# edges_from_segments = canny(all_segments_mask, 1, 1)
# edges_from_segments = cv2.cvtColor(edges_from_segments, cv2.COLOR_GRAY2BGR)


# all_segments_mask = create_segments_from_ldc_edges()

def render():
    print('rendering...')
    global all_segments_mask, img, all_possible_colors, selected_segments

    start = time.time()


    tmp_tensor_out = torch.clone(tensor_out)
    tmp_tensor_out[tmp_tensor_out >= sobel_1_thresh_slider.val] = 255
    tmp_tensor_out[tmp_tensor_out < sobel_1_thresh_slider.val] = 0
    tmp_tensor_out = np.uint8(tmp_tensor_out.detach().numpy())
    edges = tmp_tensor_out.astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(dilate_1_slider.val),int(dilate_1_slider.val)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


    # edges = sobel(img, int(sobel_1_thresh_slider.val))



    edges = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value = (255,255,255))
    # edges = cv2.bitwise_not(edges)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hiere = cv2.findContours(edges.astype(np.int32), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=lambda x: -cv2.contourArea(x))
    contours = filter(lambda x: cv2.contourArea(x) >= 20, contours)
    print(f'first pixel: {edges[1,1]}')

    colors = all_possible_colors_orig[:]

    rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    all_segments_mask = np.zeros_like(rgb_edges)
    # print(hierarchy)

    kernel_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(dilate_2_slider.val),int(dilate_2_slider.val)))

    for (i, c) in enumerate(contours):
        # print(hierarchy[0][i])
        # if hierarchy[0][i][3] == -1:

        # c = contours[198]
        # c = contours[int(dilate_3_slider.val)]

        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0])

        print(f'i: {i}')
        print(f'leftmost: {leftmost[0]} {leftmost[1]}')
        print(f'rightmost: {rightmost[0]} {rightmost[1]}')
        print(f'topmost: {topmost[0]} {topmost[1]}')
        print(f'bottommost: {bottommost[0]} {bottommost[1]}')

        # print('---')
        left = edges[leftmost[1], leftmost[0]+1]
        right = edges[rightmost[1], rightmost[0]-1]
        top = edges[topmost[1]+1, topmost[0]]
        bottom = edges[bottommost[1]-1, bottommost[0]]
        print(left)
        print(right)
        print(top)
        print(bottom)

        # if left == 255 and right == 255 and top == 255 and bottom == 255:
        #     pass
        # else:
        # cv2.drawContours(all_segments_mask, [c], 0, colors.pop(), -1)
        if left == 0 and right == 0 and top == 0 and bottom == 0:
            one_contour_drawn = np.zeros_like(rgb_edges)
            cv2.drawContours(one_contour_drawn, [c], -1, colors.pop(), -1)
            one_contour_drawn = cv2.dilate(one_contour_drawn, kernel_el)
            # all_segments_mask =  cv2.add(all_segments_mask, one_contour_drawn)
            all_segments_mask = add_non_black_to_image(all_segments_mask, one_contour_drawn)

    # all_segments_mask = cv2.dilate(all_segments_mask, kernel_el, borderType=cv2.BORDER_REPLICATE)
    # all_segments_mask = add_non_black_to_image(all_segments_mask, one_contour_drawn)

        # break

    # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # all_segments_mask = add_non_black_to_image(rgb_edges, all_segments_mask)
        # else:
        #     print('skipping')



    # mask_inv = cv2.inRange(edges, np.array([0, 0, 0]), np.array([0, 0, 0]))
    # mask_inv = cv2.bitwise_not(edges)
    # all_segments_mask = cv2.bitwise_and(all_segments_mask, all_segments_mask, mask=mask_inv)

    # _, ldc_edges_binary = cv2.threshold(ldc_edges, int(sobel_1_thresh_slider.val), 255, cv2.THRESH_BINARY_INV)
    # _, ldc_edges_binary = cv2.threshold(ldc_edges, int(sobel_1_thresh_slider.val), 255, cv2.THRESH_BINARY_INV)
    # segments_mask, colors = fill_sobel_segments(img, min_filled_pixels_per_segment=int(
    #     min_area_1_slider.val), dilate_size=int(dilate_1_slider.val), skip_step=int(skip_step_slider.val), all_possible_colors=colors1, edges=ldc_edges_binary)


    # selected_segments = flood_fill_segmented_on_touch_points(all_segments_mask)
    # propagated_image = propagate_image(img, selected_segments, added_opacity = sobel_2_thresh_slider.val)
    all_segments_mask = all_segments_mask[1:1+size[1], 1:1+size[0]]
    propagated_image = propagate_image(img, all_segments_mask, added_opacity = 127)
    # final_image = create_final_image(img, propagated_image, selected_segments)
    final_image = create_final_image(img, propagated_image, all_segments_mask)

    # # if l_mouse_is_pressed:
    # #     final_image = add_non_black_to_image(final_image, edges_from_segments)

    # final_image = do_meanshift(img, band=sobel_1_thresh_slider.val)
    # final_image = posterize(img, n=int(sobel_1_thresh_slider.val))
    # all_segments_mask = final_image

    # segments_color = get_colors_from_bigger_segments(img)

    # final_image = remove_soft_transitions_between_segments(img)
    # all_segments_mask = final_image

    # print(f'segments_color.len: {len(segments_color)}')
    # print(f'segments_color: {segments_color}')

    print(f'time tooks: {int((time.time() - start)*1000)} ms')

    # edges = sobel(img, int(sobel_1_thresh_slider.val))

    # cv2.imshow("propagated_image", propagated_image)

    # cv2.imshow("selected_segments", selected_segments)
    cv2.imshow("final_image", final_image)
    # cv2.imshow("edges_from_segments", edges_from_segments)

    cv2.imshow("img", img)
    cv2.imshow("edges", edges)
    # cv2.imshow("sobel", edges)
    # cv2.imshow("ldc_edges", ldc_edges)
    # cv2.imshow("ldc_edges_binary", ldc_edges_binary)
    # cv2.imshow("segments_mask", segments_mask)
    # cv2.imshow("flood fill", mask)
    cv2.imshow("all_segments_mask", all_segments_mask)

    fig.canvas.draw_idle()
    print('finish render')


sobel_1_thresh_slider.on_changed(sliders_on_changed)
sobel_2_thresh_slider.on_changed(sliders_on_changed)
sobel_3_thresh_slider.on_changed(sliders_on_changed)
dilate_1_slider.on_changed(sliders_on_changed)
dilate_2_slider.on_changed(sliders_on_changed)
dilate_3_slider.on_changed(sliders_on_changed)
min_area_1_slider.on_changed(sliders_on_changed)
min_area_2_slider.on_changed(sliders_on_changed)
min_area_3_slider.on_changed(sliders_on_changed)
skip_step_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')


def reset_button_on_clicked(mouse_event):
    cv2.imwrite('all_segments_mask.png', all_segments_mask)

    # touch_points.clear()


    # render()
reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color
color_radios_ax = fig.add_axes([0.025, 0.8, 0.15, 0.15])
color_radios = RadioButtons(
    color_radios_ax, ('cv2.MORPH_ELLIPSE', 'cv2.MORPH_CROSS', 'cv2.MORPH_RECT'), active=0)


def color_radios_on_clicked(label):
    global dilate_shape
    print(label)
    if label == 'cv2.MORPH_ELLIPSE':
        dilate_shape = cv2.MORPH_ELLIPSE
    elif label == 'cv2.MORPH_CROSS':
        dilate_shape = cv2.MORPH_CROSS
    elif label == 'cv2.MORPH_RECT':
        dilate_shape = cv2.MORPH_RECT
    render()

    fig.canvas.draw_idle()


color_radios.on_clicked(color_radios_on_clicked)


l_mouse_is_pressed = False
r_mouse_is_pressed = False


def add_touch_point(touch_x, touch_y):
    global touch_points

    print('x = %d, y = %d' % (touch_x, touch_y))

    size = 1
    stride = 1

    assert size % 2 == 1
    middle_index = int((size - 1) / 2)

    for y in range(0, size, stride):
        for x in range(0, size, stride):
            pixel_x = int(touch_x + x - middle_index)
            pixel_y = int(touch_y + y - middle_index)
            if pixel_x < 0 or pixel_y < 0 or pixel_x >= W or pixel_y >= H:
                continue

            touch_points.append((pixel_x, pixel_y))

    touch_points.sort()
    touch_points = list(k for k, _ in itertools.groupby(touch_points))


def on_mouse(event, x, y, _, __):
    global l_mouse_is_pressed, r_mouse_is_pressed, touch_points

    if event == cv2.EVENT_LBUTTONDOWN:
        l_mouse_is_pressed = True
        add_touch_point(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        l_mouse_is_pressed = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        r_mouse_is_pressed = True
        remove_all_points_int_this_segment(x, y)
    elif event == cv2.EVENT_RBUTTONUP:
        r_mouse_is_pressed = False
    elif event == cv2.EVENT_MBUTTONDOWN:
        # undo
        touch_points = touch_points[:-1]
    elif event == cv2.EVENT_MOUSEMOVE and l_mouse_is_pressed:
        add_touch_point(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and r_mouse_is_pressed:
        remove_all_points_int_this_segment(x, y)
    else:
        return
    render()


def remove_all_points_int_this_segment(x, y):
    global touch_points

    new_touch_points = []

    # we cannot just easily compare colors, because currently sandly not each segment has it's own color.
    mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(selected_segments, mask, seedPoint=(x, y), newVal=(
        255, 0, 0), loDiff=(1,) * 3, upDiff=(1,) * 3, flags=floodflags)
    segment_mask_under_r_touch_point = mask[1:1 + H, 1:1 + W]

    for touch_point in touch_points:
        if segment_mask_under_r_touch_point[touch_point[1], touch_point[0]] == 0:
            new_touch_points.append(touch_point)

    touch_points = new_touch_points


render()

cv2.setMouseCallback('final_image', on_mouse)

plt.show()
