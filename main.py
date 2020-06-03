import os

import ndjson
import numpy as np

from src import converter, utils

if __name__ == '__main__':

    data_dir = r"data/sample/"
    output_dir = r"output/"
    out_shape = 64
    n = 100  # -1 for all
    out_img_type = "b"  # b = binary image or g = greyscale image
    save_png = False
    png_output_dir = r"output/"

    # TODO : Should be adaptive
    line_diameter = padding = 12  # int(0.57 * out_shape)

    for data_file in os.listdir(data_dir):
        if ".ndjson" not in data_file:
            continue

        print(f"Processing {data_file}...")
        json_data = ndjson.load(open(os.path.join(data_dir, data_file)))

        vector_imgs = []
        for i, d in enumerate(json_data):
            if i > (n-1) != -1:
                break
            vector_imgs.append(d['drawing'])

        raster = converter.vector_to_raster(vector_imgs, side=out_shape, line_diameter=line_diameter, padding=padding)
        raster_imgs = np.array(raster).reshape((-1, out_shape, out_shape))

        if out_img_type == "b":
            raster_imgs = utils.binarize_images(raster_imgs)

        np.save(os.path.join(output_dir, data_file.split(".")[0]+".npy"), raster_imgs)

        if save_png:
            utils.save_images(raster_imgs, output_dir=png_output_dir, shape=(out_shape, out_shape))
