
import cv2

from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

img = cv2.imread("photo.jpg")
# img = cv2.GaussianBlur(img, (3,3), 0) 
# img = cv2.imread("photo_small.jpg")
# cv2.imshow('Original', img)
(H, W) = img.shape[:2]


def binaryThreshold(mask, thresh):
    # print("calculating thres")
    ret, thresh = cv2.threshold(mask,thresh,255,cv2.THRESH_BINARY)
    return thresh

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

dilate_shape = cv2.MORPH_ELLIPSE

def dilate(edges, size):
    dilatation_size = int(size)
    dilation_shape = cv2.MORPH_TOPHAT
    # element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
    #                                    (dilatation_size, dilatation_size))
    kernel = cv2.getStructuringElement(dilate_shape, (int(size),int(size)))

    # edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # return cv2.dilate(edges, element)

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
floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)


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
canny_min_thresh_slider = Slider(canny_min_thresh_slider_ax, 'Min thresh', 0.1, 500.0, valinit=canny_min_thresh)

canny_max_thresh_slider_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
canny_max_thresh_slider = Slider(canny_max_thresh_slider_ax, 'Max thresh', 0.1, 500.0, valinit=canny_max_thresh)

sobel_thresh_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
sobel_thresh_slider = Slider(sobel_thresh_slider_ax, 'Sobel thresh', 0.1, 500.0, valinit=sobel_thresh)

misc_slider_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
misc_slider = Slider(misc_slider_ax, 'misc slider', 0, 20, valinit=1)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    # cv2.imshow("sobel", sob)
    # cv2.imshow("canny", canny(canny_min_thresh_slider.val, canny_max_thresh_slider.val))
    render()


all_segments_mask = np.zeros((H,W,3), np.uint8)

# def flood_fill_all_areas():
#     for touch_point in touch_points:
#         print('tuoch point', touch_point[0], touch_point[1])
#         if edges[touch_point[1], touch_point[0]] != 255:
#             cv2.floodFill(edges, mask, seedPoint=touch_point, newVal=(255,0,0), loDiff=(1,)*3, upDiff=(1,)*3, flags=floodflags)
#         draw_rectangle(mask, touch_point)

last_used_color = (0,0,80)

def get_next_color():
    global last_used_color

    step = 5
    max_value = 255 - step

    if last_used_color[0] <= max_value:
        last_used_color = (last_used_color[0]+step, last_used_color[1], last_used_color[2])
    else:
        if last_used_color[1] <= max_value:
            last_used_color = (last_used_color[0], last_used_color[1]+step, last_used_color[2])
        else:
            if last_used_color[2] <= max_value:
                last_used_color = (last_used_color[0], last_used_color[1], last_used_color[2]+step)
            else:
                last_used_color = (0,255,0)

    return last_used_color


def is_within_boundaries(x, y, W, H):
    if (x < 0 or x >= W) or (y < 0 or y >= H):
        return False
    else:
        return True

def is_black(rgb_array):
    return not np.any(rgb_array)

                
def dillute_to_neighbour_if_empty(neighbour_x, neighbour_y, current_pixel):
    global W, H, all_segments_mask
    if is_within_boundaries(neighbour_x, neighbour_y, W, H) and is_black(all_segments_mask[neighbour_y, neighbour_x]):
        # all_segments_mask[neighbour_y, neighbour_x] = current_pixel
        all_segments_mask[neighbour_y, neighbour_x] = (0, 0, 255)

def render():
    global all_segments_mask

    # edges = dilate(canny(canny_min_thresh_slider.val, canny_max_thresh_slider.val), misc_slider.val)
    edges = dilate(sobel(img, sobel_thresh_slider.val), misc_slider.val)
    # edges = sobel(edges, canny_min_thresh_slider.val, depth=cv2.CV_8U)
    # mask = np.zeros((H+2,W+2),np.uint8)

    painted_areas = 0

    # for touch_point in touch_points:
    for y in range(H):
        for x in range(W):
            # print(f'processing pixel: ({x},{y})')
            # print(f'all_segments_mask[y,x]: {all_segments_mask[y,x]}')
            if edges[y, x] != 255 and is_black(all_segments_mask[y,x]):
                mask = np.zeros((H+2,W+2), np.uint8)
                cv2.floodFill(edges, mask, seedPoint=(x,y), newVal=(255,0,0), loDiff=(1,)*3, upDiff=(1,)*3, flags=floodflags)
                mask = mask[1:1+H, 1:1+W]
                # print(f'W: {W}')
                # print(f'H: {H}')
                # print(f'mask.shape: {mask.shape}')

                solid_color_image = np.zeros((H,W,3), np.uint8)
                solid_color_image[:,0:W] = get_next_color()

                # get first masked value (foreground)
                new_colored_segment = cv2.bitwise_or(solid_color_image, solid_color_image, mask=mask)
                all_segments_mask = cv2.bitwise_or(new_colored_segment, all_segments_mask)
                painted_areas += 1

    for y in range(H):
        for x in range(W):
            current_pixel = all_segments_mask[y,x]
            if not is_black(current_pixel):
                neighbours = [
                    (x-1, y-1),
                    (x, y-1),
                    (x+1, y-1),
                    (x+1, y),
                    (x+1, y+1),
                    (x, y+1),
                    (x-1, y+1),
                    (x-1, y)
                ]

                for neighbour in neighbours:
                    neighbour_x = neighbour[0]
                    neighbour_y = neighbour[1]
                    dillute_to_neighbour_if_empty(neighbour_x, neighbour_y, current_pixel),
                    

    print(f'painted areas: {painted_areas}')
    print(f'W*H: {W*H}')

    cv2.imshow("sobel", edges)
    # cv2.imshow("flood fill", mask)
    cv2.imshow("all_segments_mask", all_segments_mask)

    fig.canvas.draw_idle()


canny_min_thresh_slider.on_changed(sliders_on_changed)
canny_max_thresh_slider.on_changed(sliders_on_changed)
sobel_thresh_slider.on_changed(sliders_on_changed)
misc_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    touch_points.clear()
    render()
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


