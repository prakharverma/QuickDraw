# QuickDraw

Experiment to convert Quick Draw dataset (https://quickdraw.withgoogle.com/data) to raster and train a Convolutional Gaussian Process on it.  

## Sample Raster

![Data 1](fig/data1.png)
![Data 2](fig/data2.png)

## Steps

### 1. Create npy and/or png file

**Execute:** 
```
cd /src/data_generator
python main.py
```

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
 
**Note:** To run the sample input, first extract the `simplified_data.zip`.
 
### 2. (Optional) Tool to view data

**Execute:** 
```
cd /src/data_generator
python data_viewer.py
```

To edit parameters: ```nano data_viewer.py```
```
Parameters

npy_file_path   :  npy file path, defaults to "output/full_simplified_ambulance.npy"
img_size        :  image size in the display, defaults to (200,200)
```

### 3. Train a GP


**Execute:** 
```
cd /src/gp
python main.py
```

To edit parameters: ```nano main.py```
```
Parameters

npy_file_path       : the npy file path, defaults to "../../sample/output"
train_n             : Number of train samples from each class, -1 for all  
test_n              : Number of test samples from each class, -1 for all
minibatch_size      : Batch size
epochs              : Number of epochs
n_classes           : Number of classes
lr                  : Learning rate
n_patches           : Number of patches for training convolutional GP
patch_shape         : Patch shape
model_output_path   : Path to save the output model, defaults to "../../model/conv_gp/"
```
