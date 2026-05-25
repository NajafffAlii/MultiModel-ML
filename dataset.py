import os
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

def load_house_attributes(inputPath):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
    
    # Identify unique zipcodes and count them
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # Loop over each of the unique zipcodes and their corresponding counts
    for (zipcode, count) in zip(zipcodes, counts):
        # If the zipcode only has a few counts, we might want to group them or just keep them
        # Some PyImageSearch tutorials filter out zipcodes with less than 25 houses, let's do the same for stability
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)
            
    # Return the dataframe
    return df

def process_house_attributes(df, train, test):
    # Initialize column names for continuous data
    continuous = ["bedrooms", "bathrooms", "area"]
    
    # Perform min-max scaling each continuous feature column to the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])
    
    # One-hot encode the zipcodes
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])
    
    # Construct tabular features by concatenating continuous and categorical
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])
    
    return trainX, testX

def load_house_images(df, inputPath):
    # initialize our images array
    images = []

    # Loop over the indexes of the houses
    for i in df.index.values:
        # The id in the dataset is 1-indexed based on row number
        # e.g., the first row in HousesInfo.txt corresponds to images like 1_frontal.jpg
        # The index in pandas starts at 0, so we use i + 1
        basePath = os.path.sep.join([inputPath, "{}_{}.jpg"])
        
        # We will load 4 images and tile them into a single 64x64 image
        housePaths = [
            basePath.format(i + 1, "bathroom"),
            basePath.format(i + 1, "bedroom"),
            basePath.format(i + 1, "frontal"),
            basePath.format(i + 1, "kitchen")
        ]
        
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")

        for housePath in housePaths:
            image = cv2.imread(housePath)
            if image is not None:
                image = cv2.resize(image, (32, 32))
            else:
                image = np.zeros((32, 32, 3), dtype="uint8")
            inputImages.append(image)
        
        # Tile the 4 images
        # Top-left: bathroom, Top-right: bedroom
        # Bottom-left: frontal, Bottom-right: kitchen
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 0:32] = inputImages[2]
        outputImage[32:64, 32:64] = inputImages[3]
        
        images.append(outputImage)
        
    return np.array(images)
