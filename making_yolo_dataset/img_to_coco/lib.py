import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)
from PIL import Image  # (pip install Pillow)
from os import listdir
from os.path import isfile, join, basename, splitext
import cv2
import time


def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new(
                        '1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def get_points(poly: Polygon) -> tuple[list, list]:
    xx, yy = poly.exterior.coords.xy
    xx.append(xx[0])
    yy.append(yy[0])
    return (xx, yy)


def merge_polygon(poly: Polygon, holes: list[Polygon]) -> Polygon:
    xx, yy = get_points(poly)
    for p in holes:
        hxx, hyy = get_points(p)
        xx.extend(hxx)
        yy.extend(hyy)
        xx.append(xx[0])
        yy.append(yy[0])
    return Polygon(zip(xx, yy))


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    sub_mask = np.array(sub_mask)

    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    # contours = measure.find_contours(sub_mask, 0.5, positive_orientation='high', fully_connected='high')

    # # image = cv2.imread('anotation_masks/test.png')
    # image = cv2.imread('anotation_masks/1.pngtemp')
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # cv2.imshow("final_image", gray)
    # cv2.waitKey(3000)

    # # img = data.astype(np.uint8)
    # # Filter using contour hierarchy
    # cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # hierarchy = hierarchy[0]
    # for component in zip(cnts, hierarchy):
    #     currentContour = component[0]
    #     currentHierarchy = component[1]
    #     # x,y,w,h = cv2.boundingRect(currentContour)
    #     # Has inner contours which means it is IN
    #     if currentHierarchy[2] < 0:
    #         print('currentHierarchy[2]')
    #         # cv2.putText(image, 'IN', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
    #     # No child which means it is OUT
    #     elif currentHierarchy[3] < 0:
    #         print('currentHierarchy[3]')
    #         # cv2.putText(image, 'OUT', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            # contour[i] = (col - 1, row - 1)
            contour[i] = (col, row)

        # Make a polygon and simplify it
        # hole = Polygon([(6, 5), (255, 4), (255, 255), (8, 255)])
        # , [[(6, 5), (255, 4), (255, 255), (8, 255)]])
        poly = Polygon(contour)

        # if poly.interiors:
        #     return Polygon(list(poly.exterior.coords))
        # else:
        #     return poly
        # for interior in poly.interiors:
        #     p = Polygon(interior)
        #     print(f'interior area:{p.area}')
        #     # if p.area > eps:
        #     #     list_interiors.append(interior)

        poly = poly.simplify(0.3, preserve_topology=False)

        # print(f'poly.interiors:{len(poly.interiors)}')

        # poly = merge_polygon(poly, [hole])

        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if len(segmentation) != 0:
            segmentations.append(segmentation)

    # start = time.time()
    # if polygons[0].contains(polygons[1]):
    #     print('polygon 0 is in 1')
    # print(f'time tooks: {int((time.time() - start)*1000)} ms')

    # polygons = [merge_polygon(polygons[0], [polygons[1]])]

    # segmentations = [np.array(merge_polygon(polygons[0], [polygons[1]]).exterior.coords).ravel().tolist()]

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area,
        'segmentation': segmentations
    }

    return annotation


def get_anotated_mask_files_with_corresponding_image_ids(anotation_masks_dir, coco_dataset_dir):
    anotation_masks = []

    image_files = [(file) for file in listdir(coco_dataset_dir) if file.endswith(
        '.jpg') or file.endswith('.png') or file.endswith('.jpeg')]
    anotated_mask_files = [(file) for file in listdir(anotation_masks_dir) if file.endswith(
        '.jpg') or file.endswith('.png') or file.endswith('.jpeg')]

    assert len(image_files) == len(anotated_mask_files)

    for mask_file in anotated_mask_files:
        corresponding_image = None
        for image in image_files:
            if splitext(mask_file)[0] == splitext(image)[0]:
                corresponding_image = image
                break

        image_id = corresponding_image
        mask_full_path = f'{anotation_masks_dir}/{mask_file}'
        anotation_masks.append((image_id, mask_full_path))
    return anotation_masks
