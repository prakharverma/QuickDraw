## QuickDraw

More about the dataset: https://quickdraw.withgoogle.com/data

### Create npy or png file

Script : ```python3 main.py```

To edit parameters: ```nano main.py```
```
Parameters

data_dir        : data directory,  defaults to "data/sample/"
output_dir      : output directory, defaults to  "output/"
out_shape       : output shape of the image, defaults to 64
n = 10000       : number of images to generate, defaults to 100
out_img_type    : b = binary image or g = greyscale image, defaults to "b" 
save_png        : save png or not, defaults to False
png_output_dir  : directory to save png images, defaults to "output/"
```
 
### Tool to view data

Script: ```python3 data_viewer.py```

To edit parameters: ```nano data_viewer.py```
```
Parameters

npy_file_path   :  npy file path, defaults to "output/full_simplified_ambulance.npy"
img_size        :  image size in the display, defaults to (200,200)
```