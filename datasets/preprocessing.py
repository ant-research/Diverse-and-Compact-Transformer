import random
import math
import numpy as np
from PIL import Image
import cv2


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class RandomGrayscalePatchReplacement(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, max_attempt_num=100):
        """
        References:
        https://arxiv.org/abs/2101.08533
        https://github.com/finger-monkey/Data-Augmentation/blob/main/trans_gray.py
        """
        if random.uniform(0, 1) >= self.probability:
            return img
        img = np.array(img)
        img = img.copy()
        image_height, image_width = img.shape[:-1]
        image_area = image_height * image_width
        for _ in range(max_attempt_num):
            target_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)
            erasing_height = int(np.round(np.sqrt(target_area * aspect_ratio)))
            erasing_width = int(np.round(np.sqrt(target_area / aspect_ratio)))
            if erasing_width < image_width and erasing_height < image_height:
                starting_height = np.random.randint(0,
                                                    image_height - erasing_height)
                starting_width = np.random.randint(0, image_width - erasing_width)
                patch_in_RGB = img[starting_height:starting_height +
                                            erasing_height,
                                            starting_width:starting_width +
                                            erasing_width]
                patch_in_GRAY = cv2.cvtColor(patch_in_RGB, cv2.COLOR_RGB2GRAY)
                for index in range(3):
                    img[starting_height:starting_height + erasing_height,
                                starting_width:starting_width + erasing_width,
                                index] = patch_in_GRAY
                break
        img = Image.fromarray(img)
        return img