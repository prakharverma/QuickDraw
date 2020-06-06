import os

from skimage.filters import threshold_otsu
import imageio
import numpy as np


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


def sample_patches(kernel, x, patch_shape, n=100):
    # FIXME: Think if a better way
    random_imgs_idx = np.random.randint(0, x.shape[0], n)
    random_imgs = x[random_imgs_idx]
    patch_shape_size = patch_shape[0] * patch_shape[1]
    all_patches = np.unique(
        kernel.get_patches(random_imgs).numpy().reshape(-1, patch_shape_size), axis=0
    )
    assert all_patches.shape[0] > n
    random_idx = np.random.randint(0, all_patches.shape[0], n)
    return all_patches[random_idx]


def load_dataset(npy_file_path, train_n=-1, test_n=-1):
    x_train = y_train = x_test = y_test = None
    for i, file in enumerate(os.listdir(npy_file_path)):
        npy_file = os.path.join(npy_file_path, file)
        img_arr = np.load(npy_file)
        img_arr = img_arr.reshape((img_arr.shape[0], -1))

        random_ids = np.random.randint(0, img_arr.shape[0], train_n+test_n)
        train_random_ids = random_ids[:train_n]
        test_random_ids = random_ids[train_n:]

        x_train = img_arr[train_random_ids, :] if x_train is None else np.concatenate((x_train, img_arr[train_random_ids, :]))
        y_train = np.array([i]*train_n) if y_train is None else np.concatenate((y_train, np.array([i]*train_n)))
        x_test = img_arr[test_random_ids, :] if x_test is None else np.concatenate((x_test, img_arr[test_random_ids, :]))
        y_test = np.array([i] * test_n) if y_test is None else np.concatenate((y_test, np.array([i] * test_n)))

    return x_train, y_train, x_test, y_test
