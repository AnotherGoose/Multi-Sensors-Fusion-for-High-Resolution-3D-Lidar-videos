import cv2
import numpy as np

#https://github.com/markjay4k/Mask-RCNN-series/blob/master/visualize_cv.py

def randomColours(N):
    np.random.seed(1)
    colours = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colours

def applyMask(image, mask, colour, alpha=0.5):
    #Apply mask to image
    for n, c in enumerate(colour):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

def displayInstances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colours = randomColours(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, colour in enumerate(colours):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = applyMask(image, mask, colour)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), colour, 1)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
        )

    return image
