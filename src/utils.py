import os

from skimage.filters import threshold_otsu
import imageio


def binarize_images(img_arr: list) -> list:
    for i, img in enumerate(img_arr):
        threshold_val = threshold_otsu(img)
        img[img <= threshold_val] = 0
        img[img > threshold_val] = 255
        img_arr[i] = img

    return img_arr


def save_images(img_list: list, output_dir: str, shape: tuple):
    for i in range(len(img_list)):
        imageio.imsave(os.path.join(output_dir, str(i) + ".png"), img_list[i, :, :].reshape(shape))

    return True
