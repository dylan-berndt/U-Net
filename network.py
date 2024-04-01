
import tensorflow as tf
import keras
from keras import layers

import os
from PIL import Image
import numpy as np

import random


def loadLabels(path: str):
    file = open(path + 'class_dict.csv')
    labels = []
    for line in file.readlines()[1:]:
        data = line.split(',')
        labels.append([data[0], [int(data[1]), int(data[2]), int(data[3])]])
    return labels


def loadDataset(path: str, dataName: str, batchSize: int, classLabels: list[str, list[int]], imageSize: tuple[int, int] = None):
    imageDir = path + dataName
    labelDir = imageDir + '_labels'

    imagePaths = os.listdir(imageDir)
    labelPaths = os.listdir(labelDir)

    images = []
    labels = []

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


def recursiveUNet(parent: keras.Layer, shape: list[int], kernelSize: list[int], depth: int, filters: int = 16):
    conv = layers.Conv2D(filters, kernelSize, activation=keras.activations.relu, padding='same')(parent)
    pool = layers.MaxPool2D()(conv)

    if depth > 1:
        middle = recursiveUNet(pool, shape, kernelSize, depth - 1, filters * 2)
    else:
        middle = pool

    upSampled = layers.UpSampling2D()(middle)
    conv2 = layers.Conv2D(filters, kernelSize, activation=keras.activations.relu, padding='same')(upSampled)
    concatenated = layers.Concatenate()([conv, conv2])

    return concatenated


def createUNet(shape: list[int, None], kernelSize: list[int], depth: int, classes: int):
    inputLayer = layers.Input(shape)
    network = recursiveUNet(inputLayer, shape, kernelSize, depth)

    output = layers.Conv2D(classes, kernelSize, padding='same')(network)
    softmax = layers.Softmax()(output)

    model = keras.Model(inputs=inputLayer, outputs=softmax)

    return model


