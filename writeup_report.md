# **Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./images/model.png "Model Visualization"
[first_example]: ./images/center_1.jpg "Training data"
[recovery1]: ./images/recovery_1.jpg "Recovery Image"
[recovery2]: ./images/recovery_2.jpg "Recovery Image"
[recovery3]: ./images/recovery_3.jpg "Recovery Image"
[normal]: ./images/normal.jpg "Normal Image"
[flipped]: ./images/flipped.jpg "Flipped Image"
[distribution]: ./images/data_distribution.png "distribution"
[loss]: ./images/12th_loss_aug.png "loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utilities.py module to data augmentation
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* test.mp4 driving video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. (Autonomous driving test)

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to build model same as NVIDIA architecture and adjust the model and augment data to successfully drive.

My first step was to build a convolution neural network model similar to the NVIDIA architecture.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the the model was overfitting.

To reduce overfitting, I added the model to BatchNormalization Layer and pooling layer for reduce input size to 1/4.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added data of these spots as many times.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 51-90) consisted of convolution neural network with 5x5 3x3 filter sizes and depths between 24, 36, 48, 64.

Here is a visualization of the architecture.

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 5 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][first_example]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to behavior that back to center.
These images show what a recovery looks like starting from right side of road to center:

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

To augment the data sat, I also flipped images and angles. (Augment data when steering angle is over 1 degree or under 0.3 degree)
For example, here is an image that has then been flipped:

![alt text][normal]
![alt text][flipped]

After the collection process, I had 135118 number of data points. I then preprocessed this data by croppoing and resizing.
Following figure shows distribution of steering after augmentation.

![alt text][distribution]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the training and validation loss was under the 0.01. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Following figure shows training and validation mean squared error loss.

![alt text][loss]
