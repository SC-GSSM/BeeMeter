import numpy as np
from detection.utils.box_tools import iou

try:
    from detection.default_boxes.nms_cython.c_nms import cpu_soft_nms

    compiled = True
    print("Faster NMS is used!")
except ModuleNotFoundError:
    print("Warning: Slower NMS is used!")
    compiled = False


def compiled_nms(decoded_boxes, confidence_threshold=0.5, nms_threshold=0.5, sigma=0.5, methode='normal'):
    if methode == 'normal':
        i = 0
    elif methode == 'linear':
        i = 1
    elif methode == 'gaussian':
        i = 2
    keep = cpu_soft_nms(decoded_boxes, sigma=sigma, Nt=nms_threshold, threshold=confidence_threshold, method=i)
    return decoded_boxes[keep]


def nms_np(decoded_boxes, confidence_threshold=0.5, nms_threshold=0.5, sigma=0.5, methode='normal'):
    """
    The non-maximum-suppression algorithm is implemented. In each step the box with the highest confidence score is
    selected and all those boxes are suppressed, which overlap too much with the selected box. The overlapping is
    controlled by a threshold value.
    :param decoded_boxes: Can be a masked array or a normal array of all valid and decoded bounding boxes of shape
     (batch_size, #boxes, (class_score, xmin, ymin, xmax, ymax))
    :param nms_threshold: in mode 'normal' all boxes are suppressed, which have an IoU equal or higher than
    'nms_threshold'. Consequently setting this threshold to 1.0 will disable nms and all boxes will be returned
    (sorted by confidence)
    :return:

    Reference: https://arxiv.org/pdf/1704.04503.pdf
    """
    result = []

    while decoded_boxes.shape[0] > 0:
        # find the box with the highest confidence
        max_box_index = np.argmax(decoded_boxes[:, 0], axis=0)
        max_box = np.copy(decoded_boxes[max_box_index])
        # add the box with the highest confidence to the result
        result.append(max_box)
        # delete the found box from the box list
        decoded_boxes = np.delete(decoded_boxes, max_box_index, axis=0)
        # if there are no more boxes break
        if decoded_boxes.shape[0] == 0:
            break
        else:
            # otherwise calculate the IoU with all other boxes...
            iou_values = iou(decoded_boxes[:, 1:], np.expand_dims(max_box[1:], axis=0), box_format='corner',
                             combinations=False)
            if methode == 'normal':
                # ... and remove all boxes that are to similar to the best box
                decoded_boxes = decoded_boxes[iou_values <= nms_threshold]
            elif methode == 'linear':
                mask = iou_values < nms_threshold
                masked_array = np.ma.masked_array(iou_values, mask=mask)
                masked_array = 1 - masked_array
                decoded_boxes[..., 0] *= masked_array
            elif methode == 'gaussian':
                decoded_boxes[..., 0] *= np.exp(-(np.square(iou_values) / sigma))
    result = np.array(result)
    result = result[result[..., 0] > confidence_threshold]
    return result


def nms_linear_faster(decoded_boxes, nms_threshold=0.5):
    # Sort the boxes by the score in descending order
    order = np.argsort(decoded_boxes[:, 0])[::-1]
    result = []
    while order.size > 0:
        i = order[0]
        result.append(i)
        overlap = iou(np.expand_dims(decoded_boxes[i, 1:], axis=0), decoded_boxes[order[1:], 1:], box_format='corner',
                      combinations=False)
        new_ind = np.where(overlap <= nms_threshold)[0]
        order = order[new_ind + 1]
    return decoded_boxes[result]


def nms(**kwargs):
    if compiled:
        return compiled_nms(**kwargs)
    else:
        return nms_np(**kwargs)


if __name__ == '__main__':
    import timeit
    import matplotlib.pyplot as plt


    def generate_random_boxes(nr):
        #np.random.seed(0)
        prob = np.expand_dims(np.random.random(size=(nr,)), axis=1)
        xs = np.sort(np.random.random_integers(low=0, high=500, size=(nr, 2)), axis=1)
        ys = np.sort(np.random.random_integers(low=0, high=500, size=(nr, 2)), axis=1)
        return np.concatenate([prob, xs[:, 0:1], ys[:, 0:1], xs[:, 1:], ys[:, 1:]], axis=1).astype(np.float32)


    reps = 4
    sizes = [10, 100, 1000, 5000, 10000, 15000, 20000, 30000, 50000]
    time_np = []
    time_cy = []
    for i in sizes:
        boxes = generate_random_boxes(i)
        t1 = timeit.Timer(lambda: compiled_nms(boxes, methode="gaussian")).timeit(number=reps)
        time_cy.append(t1)
        t2 = timeit.Timer(lambda: nms_np(boxes, methode="gaussian")).timeit(number=reps)
        time_np.append(t2)
    plt.plot(sizes, time_np, label="Numpy")
    plt.plot(sizes, time_cy, label="Cython")
    plt.legend()
    plt.title("Comparing Soft-NMS (Gauß) with Numpy and Cython implementation.")
    plt.xlabel("n")
    plt.ylabel("Time")
    plt.tight_layout()
    plt.show()
