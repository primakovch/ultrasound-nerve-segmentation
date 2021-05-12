# ultrasound-nerve-segmentation
## Problem definition:
The task is to identify nerve structures of an ultrasound image. The nerve collection given specifically in the dataset is called the Brachial Plexus(BP).
## Data description:
The dataset description given in the Kaggle website(from where it was downloaded) is as follows:
1. /train/ contains the training set images, named according to subject_imageNum.tif. Every image with the same subject number comes from the same person. This folder also includes binary mask images showing the BP segmentations.
2 . /test/ contains the test set images, named according to imageNum.tif. You must predict the BP segmentation for these images and are not provided a subject number. There is no overlap between the subjects in the training and test sets.
3. train_masks.csv gives the training image masks in run-length encoded format. This is provided as a convenience to demonstrate how to turn image masks into encoded text values for submission.
sample_submission.csv shows the correct submission file format.

The complete data description as well as the download link can be found <a href = "https://www.kaggle.com/c/ultrasound-nerve-segmentation/data" target = "_blank">here.</a> 
## Solution approach:
### 1.Extracting the images
The training input image as well as the mask images are extracted from the files by writing a simple extraction code. Then, the train images and the mask are converted into numpy arrays and saved seperately into two different numpy format files "imgs_train.npy" and "imgs_mask_train.npy" respectively.
The python script for this process is not uploaded.
### 2.Preprocessing the images:
The training images are resized into (96,96) using cv2 function and the pixel values are converted to float datatype. Out of the total 5635 image,mask pairs, 5000 are used for training while the remaining are used for testing.
### 3.Model architecture and training:
The model used for training is unet model which is popular for its semantic segmentation application.
The model summary is will be given in the file "model_summary.txt"
The model is then trained for 60 epochs using Adam optimizer and dice coefficient loss as the loss function and initial learning rate set to 1e-5.

### 4.Loss function and metric:
The metric used to evaluate the model is dice coefficient and the loss function is the dice coefficient loss which is the negative of the dice coefficient.

## Performance and predition:
The model is used to predict the labeled test image. One output image sample along with the original image and the overlapped one is provided in the image directory. The learning curve is also provided in the image directory.

## Note:
1. The pretrained weights are also provided for the users. Users can simply load the pretrained model and use it for transfer learning.
2. Users are also encouraged to play with the hyperparameters to improve the model performance.
3. Users can also increase the epoch value and the input image shape if they have sufficient computing power.
4. If users want the script to extract the image(or any other queries), they can reach out to the following email: projectmacine785@gmail.com
5. code to load model:
      my_model = keras.models.load_model('path to pretrained weights', custom_objects = {"dice_coef":dice_coef, "dice_coef_loss": dice_coef_loss}, compile=True, options=None)

## References:,
1. Link to the original unet paper: <a href = "https://arxiv.org/abs/1505.04597" target = "_blank">click here</a>
2. Link to download the dataset: <a href = "https://www.kaggle.com/c/ultrasound-nerve-segmentation/data" target = "_blank">click here</a>
3. Dice coefficient article: <a href = "https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2#:~:text=3.,of%20pixels%20in%20both%20images.&text=The%20Dice%20coefficient%20is%20very%20similar%20to%20the%20IoU" target = "_blank">click here</a>



