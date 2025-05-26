#!/usr/bin/env python3

## import modules
import kagglehub
import numpy as np
from framework import (
    Model, InputLayer, ResidualBlock, FullyConnectedLayer, ReLULayer, DropoutLayer, LinearLayer, SquaredError, NormalizationLayer
)
import matplotlib.pyplot as plt
import os
import time


## define constants
KAGGLE_DATASET = 'shaikasif89/wheat-yeild'
TRAIN_TEST_SPLIT = 2/3
TERMINATE_EPOCH = 1e4
TERMINATE_RMSE = 1e-10
DIMENSIONALITY = [22, 64, 32, 16, 8, 1]
DROPOUT_PROBABILITY = 0.01
LAYER_REPEATS = 5
BATCH_SIZE = 256
SUBSET_SIZE = 5000


## import data, subset for speed, split into training and test
data_dir = kagglehub.dataset_download(KAGGLE_DATASET)
data_path = os.path.join(data_dir, os.listdir(data_dir)[0])
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

data = data[np.random.choice(data.shape[0], size=SUBSET_SIZE, replace=False)]

n_train = int(data.shape[0] * TRAIN_TEST_SPLIT)
training_indices = np.random.choice(data.shape[0], size=n_train, replace=False)
mask = np.zeros(data.shape[0], dtype=bool)
mask[training_indices] = True

X_train = data[mask, :-1]
X_test = data[~mask, :-1]
Y_train = data[mask, -1].reshape(-1, 1)
Y_test = data[~mask, -1].reshape(-1, 1)


## instantiate models
# shallow model
model_1 = Model(
    InputLayer(X_train),

    FullyConnectedLayer(DIMENSIONALITY[0], DIMENSIONALITY[1], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[1]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    FullyConnectedLayer(DIMENSIONALITY[1], DIMENSIONALITY[2], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[2]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    FullyConnectedLayer(DIMENSIONALITY[2], DIMENSIONALITY[3], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[3]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    FullyConnectedLayer(DIMENSIONALITY[3], DIMENSIONALITY[4], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[4]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    FullyConnectedLayer(DIMENSIONALITY[4], DIMENSIONALITY[5], LinearLayer),
    LinearLayer(),
    
    SquaredError()
)

# deep model without skip residuals
model_2 = Model(
    InputLayer(X_train),

    FullyConnectedLayer(DIMENSIONALITY[0], DIMENSIONALITY[1], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[1]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    *(
        layer
        for _ in range(LAYER_REPEATS)
        for layer in (
            FullyConnectedLayer(DIMENSIONALITY[1], DIMENSIONALITY[1], ReLULayer),
            NormalizationLayer(DIMENSIONALITY[1]),
            ReLULayer(),
            DropoutLayer(DROPOUT_PROBABILITY)
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[1], DIMENSIONALITY[2], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[2]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    *(
        layer
        for _ in range(LAYER_REPEATS)
        for layer in (
            FullyConnectedLayer(DIMENSIONALITY[2], DIMENSIONALITY[2], ReLULayer),
            NormalizationLayer(DIMENSIONALITY[2]),
            ReLULayer(),
            DropoutLayer(DROPOUT_PROBABILITY)
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[2], DIMENSIONALITY[3], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[3]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    *(
        layer
        for _ in range(LAYER_REPEATS)
        for layer in (
            FullyConnectedLayer(DIMENSIONALITY[3], DIMENSIONALITY[3], ReLULayer),
            NormalizationLayer(DIMENSIONALITY[3]),
            ReLULayer(),
            DropoutLayer(DROPOUT_PROBABILITY)
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[3], DIMENSIONALITY[4], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[4]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    *(
        layer
        for _ in range(LAYER_REPEATS)
        for layer in (
            FullyConnectedLayer(DIMENSIONALITY[4], DIMENSIONALITY[4], ReLULayer),
            NormalizationLayer(DIMENSIONALITY[4]),
            ReLULayer(),
            DropoutLayer(DROPOUT_PROBABILITY)
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[4], DIMENSIONALITY[5], LinearLayer),
    LinearLayer(),
    
    SquaredError()
)

# deep model with skip residuals
model_3 = Model(
    InputLayer(X_train),

    FullyConnectedLayer(DIMENSIONALITY[0], DIMENSIONALITY[1], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[1]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    ResidualBlock(
        *(
            layer
            for _ in range(LAYER_REPEATS)
            for layer in (
                FullyConnectedLayer(DIMENSIONALITY[1], DIMENSIONALITY[1], ReLULayer),
                NormalizationLayer(DIMENSIONALITY[1]),
                ReLULayer(),
                DropoutLayer(DROPOUT_PROBABILITY)
            )
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[1], DIMENSIONALITY[2], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[2]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    ResidualBlock(
        *(
            layer
            for _ in range(LAYER_REPEATS)
            for layer in (
                FullyConnectedLayer(DIMENSIONALITY[2], DIMENSIONALITY[2], ReLULayer),
                NormalizationLayer(DIMENSIONALITY[2]),
                ReLULayer(),
                DropoutLayer(DROPOUT_PROBABILITY)
            )
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[2], DIMENSIONALITY[3], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[3]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    ResidualBlock(
        *(
            layer
            for _ in range(LAYER_REPEATS)
            for layer in (
                FullyConnectedLayer(DIMENSIONALITY[3], DIMENSIONALITY[3], ReLULayer),
                NormalizationLayer(DIMENSIONALITY[3]),
                ReLULayer(),
                DropoutLayer(DROPOUT_PROBABILITY)
            )
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[3], DIMENSIONALITY[4], ReLULayer),
    NormalizationLayer(DIMENSIONALITY[4]),
    ReLULayer(),
    DropoutLayer(DROPOUT_PROBABILITY),

    ResidualBlock(
        *(
            layer
            for _ in range(LAYER_REPEATS)
            for layer in (
                FullyConnectedLayer(DIMENSIONALITY[4], DIMENSIONALITY[4], ReLULayer),
                NormalizationLayer(DIMENSIONALITY[4]),
                ReLULayer(),
                DropoutLayer(DROPOUT_PROBABILITY)
            )
        )
    ),

    FullyConnectedLayer(DIMENSIONALITY[4], DIMENSIONALITY[5], LinearLayer),
    LinearLayer(),
    
    SquaredError()
)


## train models
training_logs = []
training_times = []
for model in model_1, model_2, model_3:
    # initialize helper variables
    rmse_train = float('inf')
    delta_rmse = float('inf')
    rows = []
    start = time.perf_counter()
    while (model.getEpoch() < TERMINATE_EPOCH) and (delta_rmse > TERMINATE_RMSE):
        epoch = model.getEpoch()
        # training
        model.train(X_train, Y_train, BATCH_SIZE)
        # evaluate
        rmse_train_old = rmse_train
        rmse_train = model.eval(X_train, Y_train, "RMSE")
        rmse_test = model.eval(X_test, Y_test, "RMSE")
        # update termination criteria
        rows.append([epoch, rmse_train, rmse_test])
        delta_rmse = abs(rmse_train - rmse_train_old)
    end = time.perf_counter()
    # save results
    training_logs.append(np.array(rows))
    training_times.append(end - start)


## display results
for training_log, training_time, model_name in zip(training_logs, training_times ['Shallow Network', 'Deep Network', 'Deep Network With Skip Residuals']):
    # print error of final epoch
    print(f"{model_name}\nTime spend training model (seconds): {training_time}\nFinal RMSE of training data: {training_log[-1, 1].round(4)}\nFinal RMSE of testing data: {training_log[-1, 2].round(4)}\n")
    # plot RMSE vs epoch
    plt.plot(training_log[:, 0], training_log[:, 1], label='Training')
    plt.plot(training_log[:, 0], training_log[:, 2], label='Testing')
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()
