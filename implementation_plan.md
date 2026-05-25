# Multimodal ML – Housing Price Prediction Using Images + Tabular Data

The goal of this project is to build a multimodal deep learning model that predicts house prices by combining both structured tabular data (number of bedrooms, bathrooms, area, zipcode) and unstructured image data (photos of the house). 

## User Review Required

Please review this implementation plan. The dataset used will be the classic `Eman Ahmed's Houses Dataset`, which provides both tabular information and four images (frontal, bedroom, bathroom, kitchen) per house.
Let me know if you would like me to use a different dataset or proceed with this standard one.

## Proposed Changes

We will build the following structure:

### 1. Data Acquisition
We will clone the `Houses-dataset` repository, which contains the `Houses Dataset` folder with `HousesInfo.txt` and corresponding JPEG images.

### 2. Preprocessing Tabular Data (`dataset.py`)
- Load `HousesInfo.txt` into a Pandas DataFrame.
- Clean the data (remove rows with missing/corrupted values).
- Apply MinMax scaling to continuous features (bedrooms, bathrooms, area).
- Apply One-Hot Encoding to the categorical feature (zipcode).
- Scale the target variable (price) to improve neural network training stability (e.g., divide by maximum price).

### 3. Preprocessing Image Data (`dataset.py`)
- Load the four images per house (frontal, bedroom, bathroom, kitchen).
- Resize each image to `32x32` or `64x64`.
- Combine the four images into a single image per house (e.g., tiled 2x2 grid) or pass them as separate channels. A tiled grid approach is simple and effective.
- Normalize pixel values to `[0, 1]`.

### 4. Multimodal Model Definition (`models.py`)
- **MLP Branch**: A Multi-Layer Perceptron (Dense layers) to process the tabular features.
- **CNN Branch**: A Convolutional Neural Network (Conv2D, MaxPooling2D, Flatten, Dense) to process the tiled images.
- **Fusion**: Concatenate the final dense outputs of the MLP and CNN branches.
- **Output Head**: Add dense layers leading to a single linear output unit for regression (predicting price).

### 5. Training Pipeline (`train.py`)
- Split the dataset into training and testing sets (e.g., 80% train, 20% test).
- Compile the model using the `Adam` optimizer and `Mean Absolute Error (MAE)` or `Mean Squared Error (MSE)` as the loss function.
- Train the model using the combined inputs `[tabular_data, image_data]`.

### 6. Evaluation (`evaluate.py`)
- Evaluate the model on the test set.
- Calculate and print Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Ensure the metrics are converted back to their original scale (actual price values).

### File Structure
#### [NEW] [dataset.py](file:///n:/PYTHONN/MultiModl%20ML/dataset.py)
Contains functions to load and preprocess tabular and image data.
#### [NEW] [models.py](file:///n:/PYTHONN/MultiModl%20ML/models.py)
Contains functions to create the MLP, CNN, and combined Multimodal models.
#### [NEW] [train.py](file:///n:/PYTHONN/MultiModl%20ML/train.py)
Main script to orchestrate data loading, model training, and evaluation.
#### [NEW] [requirements.txt](file:///n:/PYTHONN/MultiModl%20ML/requirements.txt)
Dependencies: `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `opencv-python`.

## Verification Plan

### Automated Tests
- We will run the training script: `python train.py`
- We will verify that training successfully completes and that the evaluation outputs `MAE` and `RMSE` metrics.
