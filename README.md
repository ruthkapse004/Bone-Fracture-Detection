# Bone Fracture Detection using Image Processing

An Image Processing project was developed by four individuals using Python language and its standard image processing libraries. The project helps to detect cracks and fractures in bones. It performs standard image processing methods like Image Pre-processing, Edge Detection, Segmentation, Feature Extraction, Classification, etc. 
1. In pre-processing, the image is converted from RGB to Grayscale, and then different types of noise like salt & paper and Gaussian noise are removed from the image.
2. For edge detection, the Canny edge detector is used. It sharpens the edges by suppressing other pixels of the image.
3. We used the K-means clustering method to segment the image into three regions bone, skin, and background by setting the value of k=3.
4. This is the main step in the image processing. Here we used Lenet architecture. LeNet is a convolutional neural network architecture. The goal of feature extraction is to extract the important information. In image processing, features can be defined as specific characteristics of an image that can be used to distinguish it from other images.
5. The last step is classification, where the features extracted in the above step are used to detect whether the bone is fractured or not.
