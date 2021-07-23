import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from detection.utils.validation import BoundingBoxError


def convert_boxes(boxes, transformation):
    """
    There are three types of bounding box formats that can be converted into each other:
        * Corner: [xmin, ymin, xmax, ymax]
        * Center: [cx, cy, w, h]
        * MinMax: [xmin, xmax, ymin, ymax]
    :param boxes: Array of bounding boxes with shape (#boxes, 4)
    :param transformation: one conversion of ['corner2minmax', 'corner2center', 'center2minmax', 'center2corner',
                           'minmax2center', 'minmax2corner']
    :return: A copy of the Input array with transformed bounding boxes. Same shape as input.
    """
    boxes = np.array(boxes)
    result = np.copy(boxes)
    if transformation == 'corner2minmax' or transformation == 'minmax2corner':
        result[:, 1] = boxes[:, 2]
        result[:, 2] = boxes[:, 1]
    elif transformation == 'minmax2center':
        result[:, 0] = (boxes[:, 1] + boxes[:, 0]) / 2  # cx
        result[:, 1] = (boxes[:, 3] + boxes[:, 2]) / 2  # cy
        result[:, 2] = (boxes[:, 1] - boxes[:, 0])  # width
        result[:, 3] = (boxes[:, 3] - boxes[:, 2])  # height
    elif transformation == 'corner2center':
        result[:, 0] = (boxes[:, 2] + boxes[:, 0]) / 2  # cx
        result[:, 1] = (boxes[:, 3] + boxes[:, 1]) / 2  # cy
        result[:, 2] = (boxes[:, 2] - boxes[:, 0])  # width
        result[:, 3] = (boxes[:, 3] - boxes[:, 1])  # height
    elif transformation == 'center2corner':
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # xmin
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0  # ymin
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0  # xmax
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # ymax
    elif transformation == 'center2minmax':
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0  # xmin
        result[:, 1] = boxes[:, 0] + boxes[:, 2] / 2.0  # xmax
        result[:, 2] = boxes[:, 1] - boxes[:, 3] / 2.0  # ymin
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0  # ymax
    else:
        raise BoundingBoxError("The conversion '{0}' was not found".format(transformation))
    return result


def draw_box(image, boxes=None, scores=None, box_format='corner', color=(0, 0, 0), thickness=1, out=None, axis=None,
             engine="cv2", cmap=None, cvt_color=True, filename="output.jpeg"):
    """
    Draws the boxes in the given format on the image.
    :param image: The image as numpy array. If None there will be a black image of size 1000x1000 for debugging
    :param boxes: Array of bounding boxes with shape (#boxes, 4)
    :param scores: the confidence score to draw it to the box
    :param box_format: The box format. Can be 'center', 'corner' or 'minmax'
    :param color: the box color in RGB format
    :param thickness: box line thickness
    :param out: could be 'save', 'show' or None (with cv returns image, with plt returns axis)
    :param axis: if engine is matplotlib, an axis can be passed for drawing
    :param engine: use Matplotlib 'plt' or OpenCV 'cv' for drawing
    :param cmap: specifies Color map for 3d images plotted with matplotlib
    :param cvt_color: If True converts BGR => RGB
    :param filename: The output file name if out='save'
    :return: Returns the image on which the boxes are drawn
    """
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_size = 0.6
    font_thickness = 1
    if image is None:
        image = np.zeros((1000, 1000))
    if box_format == 'minmax':
        boxes = convert_boxes(boxes, 'minmax2corner')
    elif box_format == 'center':
        boxes = convert_boxes(boxes, 'center2corner')
    elif not box_format == 'corner':
        raise BoundingBoxError("The format '{0}' was not found".format(box_format))

    if engine == "plt":
        if axis:
            ax = axis
            fig = None
        else:
            fig, ax = plt.subplots(1)

        if image.ndim == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.axis('off')
            ax.imshow(image, cmap=cmap)
        ax.axis('off')
        if boxes is not None:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=thickness, edgecolor='r',
                                         facecolor='none')
                if scores is not None:
                    ax.text(x1 + 4, y1 + 4, '{:.2f}'.format(scores[i]), bbox=dict(facecolor=(1, 0, 0), alpha=0.5),
                            fontsize=9, color='white')
                ax.add_patch(rect)
        if out == 'save':
            if fig is None:
                raise ValueError("Cannot find figure on given axis.")
            fig.savefig('image_with_bboxes.png')
        elif out == 'show':
            plt.show()
        return ax
    else:
        new_image = np.array(image, copy=True)
        if axis is not None:
            print("Warning: You are using an axis but the OpenCV engine.")
        if cvt_color:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        for nr, box in enumerate(boxes):
            cv2.rectangle(new_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color,
                          thickness=thickness)
            if scores is not None:
                x, y = int(box[0]), int(box[1])
                text = str(round(scores[nr], 2))
                text_width, text_height = cv2.getTextSize(text, font, font_size, font_thickness)[0]
                cv2.putText(new_image, text, (int(x), int(y - text_height / 3)), font, font_size,
                            color, font_thickness, cv2.LINE_AA)
        if out == 'show':
            cv2.imshow('Image with boxes', new_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        elif out == 'save':
            cv2.imwrite(filename, new_image)
        return new_image


def relative2absolute(boxes, image_size, box_format='corner'):
    """
    Converts all bounding boxes from relative to absolute coordinates, depending on the image_size.
    :param boxes: Array of bounding boxes with shape (#boxes, 4) in relative coordinates or
     3d array of shape (batch, #boxes, 4)
    :param image_size: The shape of the image (width, height)
    :param box_format: The box format. Can be 'center', 'corner' or 'minmax'
    :return: A copy of the array with absolute bounding boxes. Same shape as input. Type: INT
    """
    boxes = boxes.astype(dtype=np.float, copy=True)

    if box_format == 'corner' or box_format == 'center':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif box_format == 'minmax':
        xmin, ymin, xmax, ymax = 0, 2, 1, 3
    else:
        raise BoundingBoxError("The format '{0}' was not found".format(box_format))

    boxes[..., [xmin, xmax]] *= image_size[0]
    boxes[..., [ymin, ymax]] *= image_size[1]

    return boxes.astype(dtype=np.int)


def absolute2relative(boxes, image_size, box_format='corner'):
    """
    Converts all bounding boxes from absolute to relative coordinates, depending on the image_size.
    Relative coordinates are within the interval [0, 1].
    :param boxes: Array of bounding boxes with shape (#boxes, 4) in absolute coordinates or in batches of shape
     (#batch, #boxes, 4)
    :param image_size: The shape of the image (width, height)
    :param box_format: The box format. Can be 'center', 'corner' or 'minmax'
    :return: Copy of the array with relative bounding boxes. Same shape as input. Type: FLOAT
    """
    boxes = boxes.astype(dtype=np.float, copy=True)

    if box_format == 'corner' or box_format == 'center':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif box_format == 'minmax':
        xmin, ymin, xmax, ymax = 0, 2, 1, 3
    else:
        raise BoundingBoxError("The format '{0}' was not found".format(box_format))

    boxes[..., [xmin, xmax]] /= image_size[0]
    boxes[..., [ymin, ymax]] /= image_size[1]

    return boxes


def box2center(boxes):
    """
    Calculates the center of a bounding rectangle
    :param boxes: array of bounding box of type [xmin, ymin, xmax, ymax], Shape: (#boxes, 4)
    :return: array of centers of a bounding rectangles, Shape: (#boxes, 2)
    """
    return np.rint((boxes[..., [0, 1]] + boxes[..., [2, 3]]) / 2).astype(int, copy=False)


def intersection(a, b, combinations=False):
    """
    Calculates the intersection of the boxes from a and b. The arrays need to be broadcastable, means their first shape
    need to be equal or one of them is 1. If the size of a and b is the same, then a list is returned in which the i-th
    position corresponds to the intersection of box a[i] and b[i].
    Caution: the intersection does not work with relative coordinates. The intersection with relative coordinates makes
    only sense if a ratio is taken.
    :param a: array of boxes of shape (number_boxes, 4)
    :param b: array of boxes of shape (number_boxes, 4)
    :param combinations: if True the combination of every box from a with b will be calculated
    :return: 1d array of the size of the maximum number of boxes from a or b OR if combinations is True a matrix of
     shape size_a x size_b
    """
    if combinations:
        shape_a = a.shape[0]
        shape_b = b.shape[0]
        intersections_min = np.maximum(np.tile(np.expand_dims(a[..., [0, 1]], axis=1), reps=(1, shape_b, 1)),
                                       np.tile(np.expand_dims(b[..., [0, 1]], axis=0), reps=(shape_a, 1, 1)))
        intersections_max = np.minimum(np.tile(np.expand_dims(a[..., [2, 3]], axis=1), reps=(1, shape_b, 1)),
                                       np.tile(np.expand_dims(b[..., [2, 3]], axis=0), reps=(shape_a, 1, 1)))
    else:
        intersections_min = np.maximum(a[..., [0, 1]], b[..., [0, 1]])
        intersections_max = np.minimum(a[..., [2, 3]], b[..., [2, 3]])

    side_lengths = np.maximum(0, intersections_max - intersections_min)
    return side_lengths[..., 0] * side_lengths[..., 1] if combinations else side_lengths[..., 0] * side_lengths[..., 1]


def clip_boxes(boxes, image_size, normalized, box_format='corner'):
    """
    Changes the size of the boxes so that they fit into the image.
    :param boxes: Array of boxes with shape (#boxes, 4)
    :param image_size: The size of the image
    :param box_format: The box format. Can be 'center', 'corner' or 'minmax'
    :param normalized: Must be set to true if the coordinates are normalized
    :return: A copy of the input array with clipped boxes and the same shape as the input array
    """
    boxes = np.copy(boxes)
    if box_format == 'center':
        boxes = convert_boxes(boxes, 'center2corner')
        box_format = 'corner'
    if box_format == 'corner':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif box_format == 'minmax':
        xmin, ymin, xmax, ymax = 0, 2, 1, 3
    else:
        raise BoundingBoxError("The format '{0}' was not found".format(box_format))

    boxes[:, [xmin, ymin]] = np.maximum(boxes[:, [xmin, ymin]], 0)
    if normalized:
        width, height = 1.0, 1.0
    else:
        width, height = image_size
    boxes[:, xmax] = np.minimum(boxes[:, xmax], width)
    boxes[:, ymax] = np.minimum(boxes[:, ymax], height)

    return boxes


def validate_boxes(boxes, image_size=None, min_area=None, box_format='corner', normalized=False):
    """
    Checks if the given boxes are valid.
    :param boxes: array of boxes with shape (#boxes, 4)
    :param image_size: If set check for each box if it is in the image
    :param min_area: if set check for each box if
    :param box_format: The box format. Can be 'center', 'corner' or 'minmax'
    :param normalized: must be set to true if the coordinates are normalized
    :return: 1D Boolean array of length #boxes
    """
    # check if minimum values are smaller then maximum values
    if box_format == 'center':
        boxes = convert_boxes(boxes, 'center2corner')
        box_format = 'corner'
    if box_format == 'corner':
        xmin, ymin, xmax, ymax = 0, 1, 2, 3
    elif box_format == 'minmax':
        xmin, ymin, xmax, ymax = 0, 2, 1, 3
    else:
        raise BoundingBoxError("The format '{0}' was not found".format(box_format))
    r = np.logical_and(boxes[:, xmin] < boxes[:, xmax], boxes[:, ymin] < boxes[:, ymax])

    if image_size is not None:
        if normalized:
            width, height = 1.0, 1.0
        else:
            width, height = image_size
        s = (boxes[:, xmin] >= 0) & (boxes[:, ymin] >= 0) & (boxes[:, xmax] <= width) \
            & (boxes[:, ymax] <= height)
        r = np.logical_and(r, s)

    if min_area is not None:
        # convert boxes to their areas
        areas = (boxes[:, xmax] - boxes[:, xmin]) * (boxes[:, ymax] - boxes[:, ymin])
        areas_mask = areas > min_area
        r = np.logical_and(r, areas_mask)
    return r


def iou(a, b, combinations=False, box_format='corner'):
    """
    Calculates the intersection over union of the boxes from a and b (Jaccard-Index). Values are in [0, 1].
    The output has the same behaviour as the intersection.
    :param a: array of boxes of shape (number_boxes, 4)
    :param b: array of boxes of shape (number_boxes, 4)
    :param combinations: if True the combination of every box from a with b will be calculated
    :param box_format: the format of the boxes in a and b
    :return: 1d array of the size of the maximum number of boxes from a or b OR if combinations a 2D matrix of shape
     size_a x size_b
    """
    if not box_format == 'corner':
        a = convert_boxes(a, '{0}2corner'.format(box_format))
        b = convert_boxes(b, '{0}2corner'.format(box_format))

    intersections = intersection(a, b, combinations=combinations)

    areas_1 = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    areas_2 = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    if combinations:
        shape_a = a.shape[0]
        shape_b = b.shape[0]
        areas_1 = np.tile(np.expand_dims(areas_1, axis=1), reps=(1, shape_b))
        areas_2 = np.tile(np.expand_dims(areas_2, axis=0), reps=(shape_a, 1))

    return intersections / (areas_1 + areas_2 - intersections + 1e-15)
