import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import locale

import dataset
import models

def main():
    print("[INFO] Loading house attributes...")
    inputPath = os.path.sep.join(["data_extracted", "Houses-dataset-master", "Houses Dataset", "HousesInfo.txt"])
    df = dataset.load_house_attributes(inputPath)
    
    print("[INFO] Loading house images...")
    imagesPath = os.path.sep.join(["data_extracted", "Houses-dataset-master", "Houses Dataset"])
    images = dataset.load_house_images(df, imagesPath)
    images = images / 255.0
    
    print("[INFO] Splitting dataset...")
    # partition the data into training and testing splits (75% train, 25% test)
    split = train_test_split(df, images, test_size=0.25, random_state=42)
    (trainAttrX, testAttrX, trainImagesX, testImagesX) = split
    
    # Scale price (target)
    maxPrice = trainAttrX["price"].max()
    trainY = trainAttrX["price"] / maxPrice
    testY = testAttrX["price"] / maxPrice
    
    # Process attributes (one-hot encode zipcodes, scale continuous features)
    (trainAttrX, testAttrX) = dataset.process_house_attributes(df, trainAttrX, testAttrX)
    
    print("[INFO] Creating models...")
    # MLP branch for tabular data
    mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
    
    # CNN branch for image data
    cnn = models.create_cnn(64, 64, 3, regress=False)
    
    # create the input to our final set of layers as the concatenate of both branches
    combinedInput = concatenate([mlp.output, cnn.output])
    
    # our final FC layer head will have two dense layers, the final one being our regression head
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    
    # our final model will accept tabular data on the MLP input and images on the CNN input
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    
    # compile the model
    print("[INFO] Compiling model...")
    opt = Adam(learning_rate=1e-3)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    
    # train the model
    print("[INFO] Training model...")
    model.fit(
        x=[trainAttrX, trainImagesX], y=trainY,
        validation_data=([testAttrX, testImagesX], testY),
        epochs=50, batch_size=8, verbose=1
    )
    
    # make predictions on the testing data
    print("[INFO] Evaluating model...")
    preds = model.predict([testAttrX, testImagesX])
    
    # convert predictions and true values back to original range
    preds = preds.flatten() * maxPrice
    testY_actual = testY.values * maxPrice
    
    # compute MAE and RMSE
    diff = preds - testY_actual
    abs_diff = np.abs(diff)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # compute mean absolute percentage error
    mape = np.mean(abs_diff / testY_actual) * 100
    
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    print("\n--- Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {locale.currency(mae, grouping=True)}")
    print(f"Root Mean Squared Error (RMSE): {locale.currency(rmse, grouping=True)}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print("--------------------------")

if __name__ == "__main__":
    main()
