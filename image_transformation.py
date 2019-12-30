import numpy as np
import cv2


def resize_image(frame, new_size):
    print("Resizing image to {}...".format(new_size))
    frame = cv2.resize(frame, (new_size, new_size))
    print("Done!")
    return frame


def make_background_black(frame):
    """
    Makes everything black and white, so we haven't problem with light and change colors
    """
    print("Making image black and white...")

    # Convert from RGB to Gray
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    print("Done!")
    return frame



def apply_image_transformation(frame):
    # Downsize it to reduce processing time.
    frame = resize_image(frame, 100)
    frame = make_background_black(frame)
    #frame = resize_image(frame, 30)
    return frame
