
import tensorflow as tf
import keras
from keras import layers

import os
from PIL import Image
import numpy as np

import random


# Loads the labels names and their corresponding
# color classifications from the CamVid dataset
def loadLabels(path: str):
    file = open(path + 'class_dict.csv')
    labels = []
    for line in file.readlines()[1:]:
        data = line.split(',')
        labels.append([data[0], [int(data[1]), int(data[2]), int(data[3])]])
    return labels


# Loads dataset at path. Useful paths are: train, test, val
# batchSize actually corresponds to the size of the dataset to be loaded
# classLabels should be a value returned by loadLabels
def loadDataset(path: str, dataName: str, batchSize: int, classLabels: list[str, list[int]],
                imageSize: tuple[int, int] = None, shuffle: bool = True):
    imageDir = path + dataName
    labelDir = imageDir + '_labels'

    imagePaths = os.listdir(imageDir)
    labelPaths = os.listdir(labelDir)

    images = []
    labels = []

    if shuffle:
        zipped = list(zip(imagePaths, labelPaths))
        random.shuffle(zipped)
        imagePaths, labelPaths = zip(*zipped)

    for i in range(batchSize):
        imageData = Image.open(imageDir + '/' + imagePaths[i])
        labelData = Image.open(labelDir + '/' + labelPaths[i])

        if imageSize is not None:
            imageData = imageData.resize(imageSize)
            labelData = labelData.resize(imageSize, Image.NEAREST)

        image = np.array(imageData) / 255.0

        labelData = np.array(labelData)
        label = np.zeros([image.shape[0], image.shape[1], len(classLabels)], dtype=np.uint8)

        for index, color in enumerate(classLabels):
            mask = np.all(labelData == color[1], axis=2)
            label[:, :, index] = mask

        images.append(image)
        labels.append(label)

        print(dataName, str(i + 1) + '/' + str(batchSize), end='\r')

    return np.array(images), np.array(labels)


# Define the UNet recursively using a depth value
# Uses skip connections 
def recursiveUNet(parent: keras.Layer, shape: list[int], kernelSize: list[int], depth: int, filters: int = 16):
    # Convolutional block and downsampling
    conv = layers.Conv2D(filters, kernelSize, activation=keras.activations.relu, padding='same')(parent)
    pool = layers.MaxPool2D()(conv)

    if depth > 1:
        # Define the next inner layer of the UNet
        middle = recursiveUNet(pool, shape, kernelSize, depth - 1, filters * 2)
    else:
        # Base case
        middle = pool

    # Upsampling and another convolutional block
    upSampled = layers.UpSampling2D()(middle)
    conv2 = layers.Conv2D(filters, kernelSize, activation=keras.activations.relu, padding='same')(upSampled)
    # Skip connection that allows for greater context
    concatenated = layers.Concatenate()([conv, conv2])

    return concatenated


# Create the full UNet model with final convolutional block and softmax
def createUNet(shape: list[int, None], kernelSize: list[int], depth: int, filters: int, classes: int):
    inputLayer = layers.Input(shape)
    network = recursiveUNet(inputLayer, shape, kernelSize, depth, filters)

    output = layers.Conv2D(classes, kernelSize, padding='same')(network)
    softmax = layers.Softmax()(output)

    model = keras.Model(inputs=inputLayer, outputs=softmax)

    return model


# Define the densely connected network with shape [poolSize, poolSize, 3]
def createDenseNet(shape: list[int, None], classes: int):
    # Flatten input image to allow for connection to Dense layer
    inputLayer = layers.Input(shape)
    reshaped = layers.Flatten()(inputLayer)

    # Dense layer needs to have output shape equivalent to the image size * the amount of 
    # classes due to one-hot encoding
    hidden = layers.Dense(shape[0] * shape[1] * classes)(reshaped)

    # Reshape into image
    output = layers.Reshape([shape[0], shape[1], classes])(hidden)
    
    softmax = layers.Softmax()(output)

    model = keras.Model(inputs=inputLayer, outputs=softmax)

    return model


# Splits images into tiles of size [poolSize, poolSize]
def tileImages(images: np.array, poolSize: int):
    imageSize = images.shape[1:3]
    split = []
    for i in range(len(images)):
        for y in range(0, imageSize[1], poolSize):
            for x in range(0, imageSize[0], poolSize):
                split.append(images[i, x:x + poolSize, y:y + poolSize, :])
    return np.array(split)


# Recompiles split images into their original shape
def undoTiling(tiles: np.array, shape: list[int], channels: int = 3):
    xTiles = shape[0] // tiles.shape[1]
    yTiles = shape[1] // tiles.shape[2]
    tilesPerImage = xTiles * yTiles
    images = np.zeros([len(tiles) // tilesPerImage, shape[1], shape[0], channels])
    for i in range(len(tiles)):
        imageIndex = i // tilesPerImage
        y = tiles.shape[2] * (i % yTiles)
        x = tiles.shape[1] * ((i // yTiles) % xTiles)
        # x = tiles.shape[1] * (i % xTiles)
        # y = tiles.shape[2] * ((i // xTiles) % yTiles)
        images[imageIndex, y:y + tiles.shape[2], x:x + tiles.shape[1], :] = tiles[i]
    return images
        
        




    
