import cv2
import imutils
import numpy as np
from imutils import contours


def group_width(group):
    '''Get the width of a flat line'''
    return group[:, 0].max() - group[:, 0].min()

def keep_spaced_elements(unsorted_list, threshold):
    '''
    Takes the unsorted list corresponding to the width of each flat line
    Returns a new list such that for every element i, j in the new list,
        the difference abs(i - j) >= threshold
    The indices of the elements to keep in the unsorted list are returned
    '''
    length = len(unsorted_list)
    if length == 0:
        return []
    sorted_list = sorted(range(length), key=lambda i: unsorted_list[i])
    indices = [sorted_list[0]]
    for i in range(1, length):
        diff = unsorted_list[sorted_list[i]] - unsorted_list[indices[-1]]
        if abs(diff) >= threshold:
            indices.append(sorted_list[i])
        else:
            if diff > 0:
                del indices[-1]
                indices.append(sorted_list[i])
    return indices


def keep_varying_lines(flat_lines, line_sim_thresh):
    '''
    Take the flat_lines and line similarity threshold as input
    Return all the lines whose difference is >= line_sim_thresh
    '''
    widths = [group_width(g) for g in flat_lines]
    indices_keep = keep_spaced_elements(widths, threshold=line_sim_thresh)
    # print('widths', widths, 'idx', indices_keep)
    flat_lines_filtered = [flat_lines[i] for i in indices_keep]
    # widths = sorted([group_width(g) for g in flat_lines_filtered])
    # print('widths_new', widths)
    return flat_lines_filtered

def group_points_by_y(points, y_range=5):
    """
    Groups a list of 2D points based on their y-coordinates. 
    Points whose y-coordinates are within a range of y_range are
    considered part of the same group.

    Args:
        points (list of list of int or float): A list of 2D points, 
            where each point is represented as a list
            of two coordinates, [x, y].
        y_range (int): An int used to control the range of y coordinates in a group

    Returns:
        list of list of int or float: A list of point groups, where each group is 
            represented as a list of
            points. Points within a group have y-coordinates that are 
            within a range of `y_range`.

    Raises:
        TypeError: If the input points are not a list of list of int or float.

    Example:
        >>> points = [[0, 0], [1, 2], [2, 2], [3, 3], [4, 5], [5, 5]]
        >>> group_points_by_y(points, y_range=2)
        [[[0, 0]], [[1, 2], [2, 2], [3, 3]], [[4, 5], [5, 5]]]

    """
    # sort the points by y-coordinate
    points = np.squeeze(points)
    points = points[np.argsort(points[:, 1])]
    
    groups = []
    group = []
    for i, point in enumerate(points):
        if len(group) == 0:
            group.append(point.tolist())
        else:
            # check if the y-coordinate of the current point is within a range of y_range
            _, y1 = group[0]
            y2 = point[1]
            if abs(y2 - y1) <= y_range:
                group.append(points[i].tolist())
            else:
                groups.append(group)
                group = [point.tolist()]
    
    # don't create groups with 1 point only (the last point)
    if len(group) > 1:
        groups.append(group)
    
    return groups

def lies_inside_contour(cont, x1, x2, y):
    '''
    Given two x coordinates and a y coordinate, 
    check if the horizontal line segment
    lies inside the contour `cont`
    '''
    for x in range(x1 + 1, x2):
        p = np.array([x, y], dtype='uint8')
        # return False if point does not lie inside contour
        if cv2.pointPolygonTest(cont, p, False) == -1:
            return False
    return True

def get_width_and_residuals(group, cont):
    '''
    Given a group of points that lie close to each other vertically,
    where `group` is sorted by x-coordinate
    and `cont` is the contour of the object,
    partition the group into two groups, 
    one that lies inside the contour and the other that may lie outside the contour
    to prevent making horizontal lines that cover a portion of the object that lies
    outside the contour
    '''
    # Get the mean y-coordinate
    y_mean = np.mean(group[:, 1]).astype('uint8')
    x1, y1 = group[0]
    x_max = x1
    y2 = y1
    remaining_group = None
    for i in range(1, len(group)):
        x2, y2 = group[i]
        y_mean =  (y1 + y2) // 2
        # mask errors at extremes
        if y_mean > 250:
            y_mean = 250
        if y_mean < 10:
            y_mean = 5
        if lies_inside_contour(cont, x1, x2, y_mean):
            if x2 > x_max:
                x_max = x2
        else:
            remaining_group = group[i:]
            break
    return (x1, x_max, (y1 + y2) // 2), remaining_group


def find_flat_lines(contour, img, w_min=0, w_max=float('inf'), 
                    line_sim_thresh=5, ratio=1, group_y_range=2):
    '''
    Find all flat horizontal lines inside the contour
    length L of line is between w_min and w_max
    No two lines can have length difference <= line_sim_thresh
    No two lines can have y-coordinate difference <= group_y_range
    Actual length of the line is ratio * L
    '''
    # group points by their y-coordinates
    groups = group_points_by_y(contour, y_range=group_y_range)
    # initialize the list of flat lines
    flat_lines = []
    
    # iterate over each group of points
    for i, group in enumerate(groups):
        # convert to numpy array
        group = np.array(group)
        # sort group by x-coordinate
        group = group[np.argsort(group[:, 0])]
        remaining_group = group.copy()
        while remaining_group is not None and len(remaining_group) >= 2:
            (x_min, x_max, y_mean), remaining_group = get_width_and_residuals(remaining_group, contour)
            # compute the width of the group
            width = x_max - x_min
            # add the group if its width is >= the threshold
            if w_min <= width <= w_max:
                line = np.array([[x_min, y_mean], [x_max, y_mean]])
                flat_lines.append(line)
    # remove lines with similar widths
    flat_lines = keep_varying_lines(flat_lines, line_sim_thresh)
        
    # the number of flat lines is the number of groups
    num_flat_lines = len(flat_lines)
    
    # add arrows to show the flat lines
    for i in range(num_flat_lines):
        line = np.array(flat_lines[i])
        x_min = np.min(line[:, 0]).astype('uint8')
        x_max = np.max(line[:, 0]).astype('uint8')
        y_mean = np.mean(line[:, 1]).astype('uint8')
        width = (x_max - x_min) / ratio
        width = round(width, 1)
        text = str(width)
        yloc = int(y_mean) + 10
        if yloc >= 240:
            yloc = int(y_mean) - 5
        cv2.arrowedLine(img, (x_min, y_mean), (x_max, y_mean), (0, 255, 0), 2)
        if x_min >= 230:
            x_min -= 5
        cv2.putText(img, text, (x_min, yloc), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return flat_lines, num_flat_lines, img
    
    
def draw_widths(img, mask, ratio=1, morph1=False, morph2=False,
                w_min=0, w_max=float('inf'), line_sim_thresh=0, group_y_range=1,
                draw=False):
    # threshold the mudstone
    thresh = np.where(mask == 2, 255, 0).astype(np.uint8)
    
    # optionally, perform some morphology
    morph = thresh.copy()
    big_element = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    small_element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    if morph1:
        # remove small rectangles
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, big_element, iterations=1)
        # restore connections
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, big_element, iterations=1)
        if morph2:
            # remove extraneous connections
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, small_element, iterations=1)
    # '''contour analysis'''
    # find the largest contour
    cnts = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts)

    # adjust thresholds based on ratio
    w_min *= ratio
    w_max *= ratio
    line_sim_thresh *= ratio

    anno_imgs = []
    for c in cnts:
        # copy the original image to prevent editing it
        orig = img.copy()
        h, w = orig.shape[:2]
        # minimum contour area should be >= 2% of image
        area = cv2.contourArea(c)
        perc = area / (h * w) * 100
        if perc < 2:
            continue
        cv2.drawContours(orig, [c], 0, (255,0,0), 1)
        # find the widths
        flat_lines, num_flat_lines, orig = find_flat_lines(c, orig, w_min=w_min,
                                                           w_max=w_max,
                                                           line_sim_thresh=line_sim_thresh,
                                                           ratio=ratio,
                                                           group_y_range=group_y_range)
        if draw:
            cv2.imshow("contour", orig)
            cv2.waitKey(0)
            print()
        anno_imgs.append(orig)
    return anno_imgs