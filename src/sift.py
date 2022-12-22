import cv2
import numpy as np
import os

def find_sift(img):
    """
    this method return sift descriptors for the image img

    arguments:

        ---> img is an opencv image in grayscale
    """
    sift = cv2.SIFT_create()
    features = sift.detect(img, None)
    list_features = [{"angle": feature.angle, "response": feature.response} for feature in features]
    

    return features, list_features

def apply_Lowe_test(matches, threshold):
    good = []
    try:
        for m, n in matches:
            # utiliser les valeur de m.distance et n.distance pour faire un test de Lowe
            # En ce moment, tous les matchs vont être retournés.
            if((m.distance/n.distance) < threshold):
                good.append(m)

    except ValueError:
        pass
    return good

def apply_lowe_and_cross(match_1_vers_2, match_2_vers_1, threshold):
    mutual_good = []
    for index, match in enumerate(match_1_vers_2):
        try:
            m, n = match
            if( (m.distance/n.distance) < threshold):
                target = m.queryIdx
                index_m_2  = m.trainIdx
                if(target == match_2_vers_1[index_m_2][0].trainIdx):
                    mutual_good.append(m)
        except ValueError:
            pass
    return mutual_good

def find_match_orb(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb_img  = cv2.ORB_create(500) #create opencv orb object for the image to identify
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann_img = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    #load the path of all the reference images
    list_path = os.listdir("src/images")
    #compute the keypoints for the image
    kp_img, des_img = orb_img.detectAndCompute(img, None)
    matches = []
    for path in list_path:
        abs_path = "src/images/"+path
        img_ref = cv2.imread(abs_path)
        #create orb object for the reference images
        orb_ref  = cv2.ORB_create(50) #create opencv orb object for the image to identify
        flann_ref = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        #copute the keypoints for the reference images
        kp_ref, des_ref = orb_img.detectAndCompute(img_ref, None)
        #find the matches
        matches1 = flann_img.knnMatch(des_img, des_ref, k=2)
        matches2 = flann_img.knnMatch(des_ref, des_img, k=2)
        matches.append(apply_Lowe_test(matches1, 0.7))
        # matches.append(apply_Lowe_test(matches1, 0.0001))
        score_matched = [len(x) for x in matches]
    matched_image = []

    test = dict(zip(list_path, score_matched)) 
    sorted_values = {k: v for k, v in sorted(test.items(), key=lambda item: item[1],reverse=True)}
    print(sorted_values)
