# Faceboxes++: A CPU Real-time Single-stage Face Detection and Alignment 

### 1. Introduction
Face detection and alignment are fundamental to many face applications, such as face recognition and facial expression analysis. However, the large visual variations and extreme lightings, impose greate challenges for these tasks in real world applications.
Currently, there are two main approachs for joint face detection and alignment is multi-cascade and single-stage. For multi-cascade, we can talk about MTCNN, which is famous for achieving state of the art accuracy with high performance. However, the information of previous cascade can not propagate into next cascade, MTCNN performs poorly on WIDER FACE hard test set. For single-state, Retina-Face is a new candidate, which focuses on accuracy rather than speed. Retina-Face applies the newest technique is RestNet and FPN for high accuracy for face detection. It also is strengthened by training with extra annotations .
### 2. Related work



![](https://i.imgur.com/tFyZjV5.png)


### 3. Faceboxes++

#### 3.1. Multi-task Loss
Besides the classification loss and bounding box regression loss of each anchor, we also consider the facial landmark regression loss:

$L = L_{cls}(p_{i}, p^*_{i}) + p^*_{i} L_{box}(t_{i}, t^*_{i}) + p^*_{i} L_{pts} (l_{i}, l^*_{i})$

**Face classification loss** $L_{cls}(p_{i}, p^*_{i})$, where: 
* $p_{i}$ is the prediction probability of anchor i being a face. 
* $p^*_{i}$ is 1 for positive anchor and 0 for negative anchor.

I apply softmax loss function for $L_{cls}$ cause this is the binary classification problem:



**Face box regression loss** $L_{box}(t_{i}, t^*_{i})$, where:
* $t_{i} = \{t_{x_{min}}, t_{y_{min}}, t_{x_{max}}, t_{y_{max}}\}_{i}$, the coordinates of the predicted box.
* $t^*_{i} = \{t^*_{x_{min}}, t^*_{y_{min}}, t^*_{x_{max}}, t^*_{y_{max}}\}_{i}$, ground-truth box associated with the positive anchor.

**Facial landmark regression loss** $L_{pts} (l_{i}, l^*_{i})$, where:
* $l_{i} = \{l_{x_{1}}, l_{y_{1}}, ..., l_{x_{5}}, l_{y_{5}}\}_{i}$, predicted five facial landmark regression , which is normalized based on the anchor center.
* $l^*_{i} = \{l^*_{x_{1}}, l^*_{y_{1}}, ..., l^*_{x_{5}}, l^*_{y_{5}}\}_{i}$, ground-truth facial landmark regression.

I apply smooth $L1$ loss function for both $L_{box}$ and $L_{pts}$.


#### 3.2. Different Rapidly Digested Convolution Layers

Differ from the original Faceboxes, some changes has been made in RDCL module, to archive a better performance and run faster:
* Smaller input size: the original input image size is 1024x1024x3 will be changed into 512x512x3
* Chossing suitable layer: I replace the first Pool layer by another Conv layer with stride 2, kernel size 7x7x12. I also replace the second 5x5 stride 2 Conv layer and second Pool layer by another Conv layer with stride 2, kernel size 3x3x64.
By this way, the model will work better to analyse the extra requirement, facial landmark regression. The down side is that it also increase the number of trainable parameters.


#### 3.3. Multiple Scale Convolutional Layers
#### 3.4. Anchor densification strategy


### 4. Training
#### 4.1. Datasets
#### 4.2. Data augmentation

Each training image is sequentially processed by the following data augmentation strategies:
* Color distortion: Applying some photo-metric distortions.
* Random cropping: Randomly croping from the original image
* Scale transformation: After random cropping, training image is resized to 512x512.
* Horizontal flipping: The resized image is horizontally flipped with probability of 0.5.
* Face-box filter: For all the faces without 5 facial keypoint annotations will be covered because they already don't contains meaningful information.

### 4. Experiments
#### 4.1. Datasets
The WIDER FACE dataset consists of 32,203 images and 393,703 face bounding boxes with a high degree of variability in scale, pose, expression,