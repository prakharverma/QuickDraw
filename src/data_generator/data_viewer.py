import random
from tkinter import *

import numpy as np
from PIL import Image, ImageTk


def show_imgs(ndarray_path, disp_img_dim=(64, 64)):
    img_arr = np.load(ndarray_path)

    def read_image():
        random_img_id = random.randint(0, img_arr.shape[0]-1)
        img = ImageTk.PhotoImage(image=Image.fromarray(img_arr[random_img_id, :, :]).resize(disp_img_dim))
        canvas.itemconfig(p, image=img)
        canvas.mainloop()

    root = Tk()
    canvas = Canvas(root, width=disp_img_dim[0]+20, height=disp_img_dim[1]+20)
    canvas.pack()
    next_image = Button(root, command=read_image, text="Next image", width=17, default=ACTIVE)
    next_image.pack()

    first_img = ImageTk.PhotoImage(image=Image.fromarray(img_arr[0, :, :]).resize(disp_img_dim))
    p = canvas.create_image(20, 20, anchor=NW, image=first_img)

    mainloop()


if __name__ == '__main__':
    npy_file_path = r"../../sample/output/full_simplified_ambulance.npy"
    img_size = (200, 200)
    show_imgs(npy_file_path, img_size)
