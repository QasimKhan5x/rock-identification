import cv2
import imutils
from imutils import perspective, contours
from google.colab.patches import cv2_imshow


def get_scale(img, mask):
    # get the foreground (0 corresponds to scale)
    _, scale_candidates = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
    # find the largest contour
    cnts = cv2.findContours(scale_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-righ
    (cnts, _) = contours.sort_contours(cnts)
    # get the largest contour using its area
    cont_max = max(cnts, key = cv2.contourArea)
    # find the bounding box of the contour
    x, y, w, h = cv2.boundingRect(cont_max)
    scale = img[y:y+h, x:x+w].copy()
    return scale

def get_contours(th, num_closing_iterations, element):
    '''
    horizontal closing and erosion followed by canny edge detection 
    followed by contour finding
    '''
    # close the gaps between each black box
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, h_element, iterations=num_closing_iterations)
    # erode horizontally to get original width of strip
    morph = cv2.erode(morph, h_element)
    # get edges
    edges = cv2.Canny(morph, 100, 200)
    # get external contours
    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def get_pixel_length_ratio(gray_img, mask, num_closing_iterations=2, actual_width=15, draw=False):
    """
    Calculates the pixel length ratio of a rectangular, yellowish scale object in a grayscale image.

    Parameters
    ----------
    gray_img : numpy.ndarray
        A grayscale image as a numpy array of shape (H, W).
    mask : numpy.ndarray
        A mask indicating the location of the scale object in the grayscale image, as a numpy array of shape (H, W).
    num_closing_iterations : int, optional
        Number of iterations to perform during closing of boxes in the strip.
    actual_width : int, optional
        The actual width of the scale object in centimeters. Default value is 15.
    actual_width : bool, optional
        Whether to draw the contour on the image for debugging purposes

    Returns
    -------
    float
        The ratio of the width of the scale object in pixels to its actual width in centimeters.

    Notes
    -----
    This function assumes that the grayscale image and mask have the same dimensions.
    The input values are expected to be in the correct range (0-2 for mask).
    """
    '''extract the scale from the mask'''
    # crop the mask to get roi for scale
    scale = get_scale(gray_img, mask)
    '''extract the boxes from the roi'''
    # blur to remove noise
    # blur = cv2.GaussianBlur(scale,(5,5),0)
    # blurring degrades thresholding so it is removed
    # threshold to get the black bloxes
    _, th = cv2.threshold(scale, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # horizontal structuring element
    h_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    cnts = get_contours(th, num_closing_iterations, element=h_element)
    i = 0
    while len(cnts) > 6:
        i += 1
        cnts = get_contours(th, num_closing_iterations + i, element=h_element)
        # function failed
        if i > 5:
            return -1
    # sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        width_perc = round(w / scale.shape[1] * 100)
        area = cv2.contourArea(c)
        area_perc = area / (scale.shape[0] * scale.shape[1]) * 100
        area_perc = round(area_perc, 2)
        aspect_ratio = round(float(w) / h, 2)
        rect_area = w*h
        extent = round(float(area) / rect_area, 2)
        if width_perc < 65 or width_perc >= 99 or area_perc == 0 or aspect_ratio < 6:
            continue
        if draw:
            orig = scale.copy()
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box).astype('int')
            cv2.drawContours(orig, [box], 0, (0, 255, 0), 1)
            cv2_imshow(orig)
        return round(w / actual_width, 1)
    return -2
