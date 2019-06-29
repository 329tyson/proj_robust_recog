import cv2

from random import random


def get_image(imagepath):
    return cv2.imread(imagepath, cv2.IMREAD_COLOR)


def bgr_to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])


def vertical_flip(image, prob):
    if random() > prob:
        return cv2.flip(image, 1)
    return image


def horizontal_flip(image, prob):
    if random() > prob:
        return cv2.flip(image, 0)
    return image


def crop_to_bounding_box(image, bbox: tuple):
    print(bbox)
    starting_x, starting_y, width, height = bbox
    return image[starting_y:starting_y + height, starting_x:starting_x + width]


def crop_to_points(image, left_top: tuple, right_bottom: tuple):
    starting_x, starting_y = left_top
    end_x, end_y = right_bottom
    return image[starting_y: end_y, starting_x: end_x]
