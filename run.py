
import cv2

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import random
import time


from lib import *


img = cv2.imread("photo.jpg")
# img = cv2.GaussianBlur(img, (3,3), 0) 
# img = cv2.imread("photo_small.jpg")
cv2.imshow('Original', img)
(H, W) = img.shape[:2]


all_possible_colors_orig = generate_unique_colors(W, H)
all_possible_colors = all_possible_colors_orig[:]


def hed():
    # print("calculating hed")
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H), swapRB=False, crop=False)
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    # return hed
    return binaryThreshold(hed)


dilate_shape = cv2.MORPH_ELLIPSE


def canny(min, max):
    return cv2.Canny(img, min, max, apertureSize=3, L2gradient=True)


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
    mask[point[1]-1, point[0]-1] = 255
    mask[point[1]-1, point[0]] = 255
    mask[point[1]-1, point[0]+1] = 255

    mask[point[1], point[0]-1] = 255
    mask[point[1], point[0]] = 255
    mask[point[1], point[0]+1] = 255

    mask[point[1]+1, point[0]-1] = 255
    mask[point[1]+1, point[0]] = 255
    mask[point[1]+1, point[0]+1] = 255


canny_min_thresh_slider_ax  = fig.add_axes([0.25, 0.25, 0.65, 0.03])
canny_min_thresh_slider = Slider(canny_min_thresh_slider_ax, 'Min thresh', 0.1, 300.0, valinit=canny_min_thresh)

canny_max_thresh_slider_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
canny_max_thresh_slider = Slider(canny_max_thresh_slider_ax, 'Max thresh', 0.1, 300.0, valinit=canny_max_thresh)

sobel_thresh_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
sobel_thresh_slider = Slider(sobel_thresh_slider_ax, 'Sobel thresh', 0.1, 300.0, valinit=sobel_thresh)

misc_slider_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
misc_slider = Slider(misc_slider_ax, 'misc slider', 0, 20, valinit=5)

b_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03])
b_slider = Slider(b_slider_ax, 'B', 0, 10, valinit=2)

c_slider_ax  = fig.add_axes([0.25, 0.0, 0.65, 0.03])
c_slider = Slider(c_slider_ax, 'C', 0, 15000, valinit=1000)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    # cv2.imshow("sobel", sob)
    # cv2.imshow("canny", canny(canny_min_thresh_slider.val, canny_max_thresh_slider.val))
    render()


all_segments_mask = np.zeros((H,W,3), np.uint8)






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
            def get_neighbours(x,y):
                return [
                    (x-1, y-1),
                    (x, y-1),
                    (x+1, y-1),
                    (x-1, y)
                ]
            x_range = list(range(W))
            y_range = list(range(H))
            color = (0,0,255)
        else:
            print('going from bottom right')
            def get_neighbours(x,y):
                return [
                    (x+1, y),
                    (x+1, y+1),
                    (x, y+1),
                    (x-1, y+1)
                ]
            x_range = list(reversed(range(W)))
            y_range = list(reversed(range(H)))
            color = (255,0,0)

        for y in y_range:
            for x in x_range:
                current_pixel = all_segments_mask[y,x]
                if is_black(current_pixel):
                    black_count += 1
                else:
                    for neighbour in get_neighbours(x,y):
                        was_diluted = dillute_to_neighbour_if_empty(neighbour_x=neighbour[0], neighbour_y=neighbour[1], current_pixel=current_pixel, col=color)
                        if was_diluted == True:
                            diluted_count += 1

        print(f'iteration: {i}, black_count: {black_count}, diluted_count: {diluted_count}')
        if black_count == diluted_count:
            break

        i += 1

def flood_fill_on_touch_points(edges):
    global all_segments_mask
    for touch_point in touch_points:
        if edges[touch_point[1], touch_point[0]] != 255:
            mask = np.zeros((H+2,W+2),np.uint8)
            cv2.floodFill(edges, mask, seedPoint=touch_point, newVal=(255,0,0), loDiff=(1,)*3, upDiff=(1,)*3, flags=floodflags)
            mask = mask[1:1+H, 1:1+W]

            solid_color_image = np.zeros((H,W,3), np.uint8)
            solid_color_image[:,0:W] = all_possible_colors[touch_point[0]+touch_point[1]]
            new_colored_segment = cv2.bitwise_or(solid_color_image, solid_color_image, mask=mask)
            all_segments_mask = cv2.bitwise_or(new_colored_segment, all_segments_mask)


def render():
    print('rendering...')
    global all_segments_mask, img, all_possible_colors


    all_segments_mask = np.zeros((H,W,3), np.uint8)

    
    # flood_fill_on_touch_points(edges)



    start = time.time()
    # # for touch_point in touch_points:
    step = int(misc_slider.val)

    segments_mask, all_possible_colors =  fill_sobel_segments(img, sobel_thresh=int(sobel_thresh_slider.val), min_filled_pixels_per_segment=c_slider.val, dilate_size=2, skip_step=8, all_possible_colors=all_possible_colors)

    all_segments_mask = cv2.add(all_segments_mask, segments_mask)

    segments_mask, all_possible_colors =  fill_sobel_segments(img, sobel_thresh=15, min_filled_pixels_per_segment=433, dilate_size=2, skip_step=8, all_possible_colors=all_possible_colors)

    all_segments_mask = add_non_black_to_image(bg_image=all_segments_mask, fg_image=segments_mask)

    print(f'time tooks: {int((time.time() - start)*1000)} ms')

    # # dillute_segments()

    # # print(f'painted areas: {painted_areas}')
    # # print(f'W*H: {W*H}')



    # cv2.imshow("sobel", edges)
    # cv2.imshow("flood fill", mask)
    cv2.imshow("all_segments_mask", all_segments_mask)
    


    # cv2.imshow("all_segments_mask edges", sobel(all_segments_mask, 1))

    # cv2.imwrite('all_segments_mask.png', all_segments_mask)

    fig.canvas.draw_idle()
    print('finish render')


canny_min_thresh_slider.on_changed(sliders_on_changed)
canny_max_thresh_slider.on_changed(sliders_on_changed)
sobel_thresh_slider.on_changed(sliders_on_changed)
misc_slider.on_changed(sliders_on_changed)
b_slider.on_changed(sliders_on_changed)
c_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    cv2.imwrite('all_segments_mask.png', all_segments_mask)

    # touch_points.clear()
    # render()
reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color
color_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15] )
color_radios = RadioButtons(color_radios_ax, ('cv2.MORPH_ELLIPSE', 'cv2.MORPH_CROSS','cv2.MORPH_RECT'), active=0)
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

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d'%(x, y))
        touch_points.append((x, y))
        render()


render()

cv2.setMouseCallback('sobel', on_mouse)

plt.show()



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