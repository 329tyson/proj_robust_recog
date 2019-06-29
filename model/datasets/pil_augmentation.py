from PIL import Image


def get_image(imagepath):
    im = Image.open(imagepath, "r")
    im = im.convert("RGB")
    return im


def crop_to_bounding_box(image, bbox: tuple):
    x, y, w, h = bbox
    w = w + x
    h = y + h
    bbox = (x, y, w, h)
    cropped_image = image.crop(bbox)
    return cropped_image
