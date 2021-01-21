import os
import sys
import time
import math
import argparse
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10, mnist
import numpy as np
from wann_src.ind import *



def main(args):
    discriminator = keras.models.load_model(args.discfile)
    wvec, avec, wkey = importNet(args.wanfile)
    for _ in range(1):
        inpatterns = np.random.normal(size=(500, 20))
        def sigmoid(z):
            return 1/(1+np.exp(-z))
        if args.mnist <= 0:
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            outs = act(wvec, avec, 20, 32*32*3, inpatterns)
            wanimgs = np.reshape(sigmoid(outs), (-1,32,32,3))
            realinds = np.random.choice(np.arange(len(X_train)), 500)
            realimgs = X_train[realinds, :, :, :] / 255
            X = np.zeros((1000, 32,32,3))
            X[:500, :, :, :] = wanimgs
            X[500:, :, :, :] = realimgs
        elif args.mnist >= 1:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            outs = act(wvec, avec, 20, 28*28, inpatterns)
            wanimgs = np.reshape(sigmoid(outs), (-1,28, 28, 1))
            realinds = np.random.choice(np.arange(len(X_train)), 500)
            realimgs = X_train[realinds, :, :, np.newaxis] / 255
            X = np.zeros((1000, 28,28,1))
            X[:500, :, :, :] = wanimgs
            X[500:, :, :, :] = realimgs
        Y = np.zeros(1000)
        Y[:500] = 0
        Y[500:] = 1

        discriminator.train_on_batch(X, Y)

    discriminator.save(args.discout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Retrain WAGAN discirminator."))

    parser.add_argument('-d', '--discfile', type=str,\
            help='pretrained discriminator file', default='domain/gan_discriminator_0.h5')

    parser.add_argument('-w', '--wanfile', type=str,\
            help='weight-agnostic neural net to use in training of new discriminator', default='log/test_best.out')

    parser.add_argument('-o', '--discout', type=str,\
            help='file to write updated discriminator to', default='domain/gan_discriminator_1.h5')

    parser.add_argument('-m', '--mnist', type=int,\
            help='update mnist discriminators (0/1)', default=0)

    args = parser.parse_args()

    main(args)
        
    
