import cv2
import os
import sift
import json
import numpy as np
import math


#get the filename of every image
images_paths  = os.listdir("src/images")
# #build empty list to contains all the features of the images 
# features = {}
# features_sum = {}
# for im_path in images_paths:
#     print(im_path)
#     img = cv2.imread("src/images/"+im_path)
#     grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     feature, list_features = sift.find_sift(grayscale)
#     grayscale = cv2.drawKeypoints(grayscale, feature, grayscale)
#     cv2.imwrite("src/sifted_images/"+im_path, grayscale)
#     features[im_path] = list_features

# with open("src/sift_features.json", "w") as file:
#     json.dump(features, file)

features = {}
with open("src\sift_features.json", "r") as sift_keypoints:
    features = json.load(sift_keypoints)

images_paths_bis = images_paths
####test du matching des images
for i in range(10):
    #select randomly an image
    index = np.random.randint(len(images_paths_bis))
    print(images_paths_bis[index])
    img_test = cv2.imread("src/images/"+images_paths_bis[index])
    #on retire cette image de la liste afin de ne plus la reprendre
    del images_paths_bis[index]
    sift.find_match_orb(img_test)
    print("_______________________________________________________")
        

# print(features)