# Bone Fracture Detection using Image Processing
Aug 2022 - Dec 2022

An Image Processing based project developed by group of four individuals using Python language and it's standard image processing libraries. The project helps to detect cracks and fractures in bones. It performs some standard image processing methods like Image Pre-processing, Edge Detection, Segmentation, Feature Extraction, Classification etc. 
1. In pre-processing the image is converted from RGB to Gray scale and then different types of noises like salt & paper noise and gaussian noise are removed from the image.
2. For edge detection canny edge detector is used. It sharpens the edges by suppressing other pixels of the image.
3. We used K-means clustering method to segment the image into three regions bone, skin and background by setting the value of k=3.
4. This is the main step in the image processing. Here we used Lenet architecture. LeNet is a convolutional neural network architecture. The goal of feature extraction is to extract the important information. In image processing, features can be defined as specific characteristics of an image that can be used to distinguish it from other images.
5. Last step is classification, where the features extracted in above step are used to detect whether the bone is fractured or not.
