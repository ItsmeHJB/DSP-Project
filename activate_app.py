from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import gzip
import json
import keras
# import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import os
import random as rand
import re
import sys
import tensorflow as tf

from flask import Flask, render_template, request
from importlib import reload
from keras import backend as K
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from activate_config import Options
from PIL import Image

oo = Options()

app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TF warnings when run.
sys.setrecursionlimit(2000)

# datagen = ImageDataGenerator(
# rotation_range=22.5,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=[-0.2, +0.2])

datagen = ImageDataGenerator(
    # rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1)


def select_starting_point(pcad, use_random=False):
    index = -1
    do_not_use_random = True

    if use_random:
        index = int(np.random.random() * oo.pca_distances.shape[0])
    else:
        temp_pca_distances = oo.pca_distances[:, 0:2]
        mean_value = np.mean(temp_pca_distances, axis=0)
        sum_value = np.sum(np.abs(temp_pca_distances - mean_value), axis=1)
        index = np.argmin(sum_value)
    print("Selected index:", index)
    oo.point_select.append(oo.pca_distances[index, :])
    this_index = oo.pca_distances[index, 0]
    oo.point_select = np.array(oo.point_select)
    pcad = np.delete(pcad, this_index, 0)
    return oo.point_select, pcad, this_index


def select_point_based_on_distance(ps, pcad):
    # we need to measure distance between selected points and all 
    # available points to decide which one should be considered next
    # print ("PCA shape: ", pcad.shape)
    # print ("PS shape: ", ps.shape)
    all_distances = []
    for i in range(ps.shape[0]):
        dist = np.sqrt(((ps[i, 1] - pcad[:, 1]) ** 2) + ((ps[i, 2] - pcad[:, 2]) ** 2))
        all_distances.append(dist)
    all_distances = np.array(all_distances)
    all_distances = np.min(all_distances, axis=0)
    index = np.argmax(all_distances)
    ps = np.append(ps, pcad[index, :].reshape(1, ps.shape[1]), axis=0)
    this_index = pcad[index, 0]

    pcad = np.delete(pcad, index, 0)
    return ps, pcad, this_index


def select_point_based_on_random(ps, pcad):
    # we need to measure distance between selected points and all 
    # available points to decide which one should be considered next
    # print ("PCA shape: ", pcad.shape)
    # print ("PS shape: ", ps.shape)
    index = int(np.random.random() * pcad.shape[0])
    ps = np.append(ps, pcad[index, :].reshape(1, ps.shape[1]), axis=0)
    this_index = pcad[index, 0]
    pcad = np.delete(pcad, index, 0)
    return ps, pcad, this_index


def select_point(ps, pcad, index):
    # we need to measure distance between selected points and all 
    # available points to decide which one should be considered next
    # print ("PCA shape: ", pcad.shape)
    # print ("PS shape: ", ps.shape)
    # ps = np.array(ps)

    if len(ps) == 0:
        ps = pcad[index, :].reshape(1, -1)
    else:
        ps = np.append(ps, pcad[index, :].reshape(1, ps.shape[1]), axis=0)

    print("PS:")
    print(ps)

    this_index = pcad[index, 0]
    pcad = np.delete(pcad, index, 0)
    return ps, pcad, this_index


#### THIS AND NEXT FUNCTION ARE TO SUPPORT MEASURING DISTANCE BASED ON FULL DATASET RATHER
#### THAN JUST 2 DIMENSIONAL PCA
def select_starting_point_for_full_image(pcad, use_random=False):
    index = -1

    if use_random:
        index = int(np.random.random() * oo.normal_distances.shape[0])
    else:
        temp_normal_distances = oo.normal_distances[:, 1:]
        mean_value = np.mean(temp_normal_distances, axis=0)
        sum_value = np.sum(np.abs(temp_normal_distances - mean_value), axis=1)
        index = np.argmin(sum_value)

    print("Selected index:", index)

    oo.point_select.append(oo.normal_distances[index, :])
    this_index = oo.normal_distances[index, 0]
    oo.point_select = np.array(oo.point_select)
    pcad = np.delete(pcad, index, 0)
    print(oo.point_select)
    print(this_index)
    return oo.point_select, pcad, this_index


def select_point_based_on_distance_for_full_image(ps, pcad):
    print("PCA shape: ", pcad.shape)
    print("PS shape: ", ps.shape)
    all_distances = []
    for i in range(ps.shape[0]):
        dist = 0
        for j in range(1, ps.shape[1]):
            dist += ((ps[i, j] - pcad[:, j]) ** 2)
        dist = np.sqrt(dist)
        all_distances.append(dist)
    all_distances = np.array(all_distances)
    all_distances = np.min(all_distances, axis=0)
    index = np.argmax(all_distances)
    ps = np.append(oo.point_select, pcad[index, :].reshape(1, ps.shape[1]), axis=0)
    this_index = pcad[index, 0]
    pcad = np.delete(pcad, index, 0)
    return ps, pcad, this_index


def select_point_based_on_random_for_full_image(ps, pcad):
    print("PCA shape: ", pcad.shape)
    print("PS shape: ", ps.shape)
    index = int(np.random.random() * pcad.shape[0])
    ps = np.append(oo.point_select, pcad[index, :].reshape(1, ps.shape[1]), axis=0)
    this_index = pcad[index, 0]
    pcad = np.delete(pcad, index, 0)
    return ps, pcad, this_index


def getDistanceTrainBatch(size):
    sample_ref = []
    counter = 0
    while (counter < size):
        if oo.batch_total == 0 and counter == 0:
            if oo.use_dimensional_reduction_mode:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_starting_point(
                    oo.pca_distances_working_copy)
            else:
                oo.point_select, oo.normal_distances_working_copy, this_index = select_starting_point_for_full_image(
                    oo.normal_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref.append(sample)
                counter += 1
        else:
            if oo.use_dimensional_reduction_mode:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point_based_on_distance(
                    oo.point_select, oo.pca_distances_working_copy)
            else:
                oo.point_select, oo.normal_distances_working_copy, this_index = select_point_based_on_distance_for_full_image(
                    oo.point_select, oo.normal_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref.append(sample)
                counter += 1

    print("What's in sample_ref", sample_ref)
    if oo.train_step_log > 0:
        if oo.use_convnet == 'keraslogreg':
            node_data = keras_current_model_prediction(sample_ref)
            return node_data
        elif oo.use_convnet == 'kerasconvnet':
            node_data = keras_current_convnet_model_prediction(sample_ref)
            return node_data
    else:
        return sample_ref


def getRandomTrainBatch(size):
    sample_ref = []
    counter = 0
    while (counter < size):
        if oo.batch_total == 0 and counter == 0:
            if oo.use_dimensional_reduction_mode:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_starting_point(
                    oo.pca_distances_working_copy, use_random=True)
            else:
                oo.point_select, oo.normal_distances_working_copy, this_index = select_starting_point_for_full_image(
                    oo.normal_distances_working_copy, use_random=True)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref.append(sample)
                counter += 1
        else:
            if oo.use_dimensional_reduction_mode:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point_based_on_random(
                    oo.point_select, oo.pca_distances_working_copy)
            else:
                oo.point_select, oo.normal_distances_working_copy, this_index = select_point_based_on_random_for_full_image(
                    oo.point_select, oo.normal_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref.append(sample)
                counter += 1

    print("What's in sample_ref", sample_ref)
    if oo.train_step_log > 0:
        if oo.use_convnet == 'keraslogreg':
            node_data = keras_current_model_prediction(sample_ref)
            return node_data
        elif oo.use_convnet == 'kerasconvnet':
            node_data = keras_current_convnet_model_prediction(sample_ref)
            return node_data
    else:
        return sample_ref


def getConfidenceRandomTrainBatch(size, method_of_conf_select='least_confidence'):
    sample_ref = []
    sample_ref_temp = []

    random_size = 10  # size

    if oo.batch_total == 0:
        counter = 0
        while (counter < size):
            if counter == 0:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_starting_point(
                    oo.pca_distances_working_copy, use_random=True)
            else:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point_based_on_random(
                    oo.point_select, oo.pca_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref.append(sample)
                counter += 1
        return sample_ref

    else:
        temp_point_select = oo.point_select
        temp_pca_distances_working_copy = oo.pca_distances_working_copy
        counter = 0
        while (counter < size * random_size):
            temp_point_select, temp_pca_distances_working_copy, this_index = select_point_based_on_random(
                temp_point_select, temp_pca_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref_temp.append(sample)
                counter += 1

        if oo.use_convnet == 'keraslogreg':
            node_data = keras_current_model_prediction(sample_ref_temp)
        elif oo.use_convnet == 'kerasconvnet':
            node_data = keras_current_convnet_model_prediction(sample_ref_temp)

        conf_values = []
        image_values = []
        nd = []

        if method_of_conf_select == 'least_confidence':
            for n in range(len(node_data)):
                # print (n)
                # print ("node_data:", node_data[n])
                # print ("predictions:", node_data[n]['predictions'])
                print("confidence:", node_data[n]['confidence'])
                conf_values.append(node_data[n]['confidence'])
            conf_values = np.array(conf_values)
            indexes = conf_values.argsort()[:size]
            print("indexes:", indexes)
            for i in range(len(indexes)):
                obj = node_data[indexes[i]]
                nd.append(obj)
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                                          oo.pca_distances_working_copy,
                                                                                          obj['id'])
        elif method_of_conf_select == 'marginal_confidence':
            for n in range(len(node_data)):
                preds = node_data[n]['predictions']
                preds = np.sort(preds)
                print("Predicted values: ", preds)
                marginal = np.abs(preds[len(preds) - 1] - preds[len(preds) - 2])
                print("Marginal value: ", marginal)
                # print (n)
                # print ("node_data:", node_data[n])
                # print ("predictions:", node_data[n]['predictions'])
                # print ("confidence:", node_data[n]['confidence'])
                conf_values.append(marginal)

            conf_values = np.array(conf_values)
            indexes = conf_values.argsort()[:]
            indexes = np.flip(indexes, 0)
            indexes = indexes[:size]
            print("indexes:", indexes)
            for i in range(len(indexes)):
                obj = node_data[indexes[i]]
                nd.append(obj)
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                                          oo.pca_distances_working_copy,
                                                                                          obj['id'])
        elif method_of_conf_select == 'entropy_confidence':
            for n in range(len(node_data)):
                print(n)

                entropy = sp.stats.entropy(node_data[n]['predictions'])
                # print ("node_data:", node_data[n])
                print("predictions:", node_data[n]['predictions'])
                print("entropy:", entropy)
                # print ("confidence:", node_data[n]['confidence'])
                conf_values.append(entropy)

            conf_values = np.array(conf_values)
            indexes = conf_values.argsort()[:]
            indexes = np.flip(indexes, 0)
            indexes = indexes[:size]

            print("indexes:", indexes)
            for i in range(len(indexes)):
                obj = node_data[indexes[i]]
                nd.append(obj)
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                                          oo.pca_distances_working_copy,
                                                                                          obj['id'])
        return nd


def getConfidenceDistanceTrainBatch(size, method_of_conf_select='least_confidence'):
    sample_ref = []
    sample_ref_temp = []
    distance_size = 10  # size
    if oo.batch_total == 0:
        counter = 0
        while (counter < size):
            if counter == 0:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_starting_point(
                    oo.pca_distances_working_copy, use_random=True)
            else:
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point_based_on_random(
                    oo.point_select, oo.pca_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref.append(sample)
                counter += 1
        return sample_ref

    else:
        temp_point_select = oo.point_select
        temp_pca_distances_working_copy = oo.pca_distances_working_copy
        counter = 0
        while (counter < size * distance_size):
            temp_point_select, temp_pca_distances_working_copy, this_index = select_point_based_on_distance(
                temp_point_select, temp_pca_distances_working_copy)
            sample = 'MNIST_image_' + str(int(this_index)) + '.png'
            if sample not in oo.all_selected_filenames:
                sample_ref_temp.append(sample)
                counter += 1
        # print ("oo.point_select.shape: ", oo.point_select.shape)
        # print ("oo.pca_distances_working_copy: ", oo.pca_distances_working_copy.shape)
        # print ("temp_point_select.shape: ", temp_point_select.shape)
        # print ("temp_pca_distances_working_copy: ", temp_pca_distances_working_copy.shape)
        if oo.use_convnet == 'keraslogreg':
            node_data = keras_current_model_prediction(sample_ref_temp)
        elif oo.use_convnet == 'kerasconvnet':
            node_data = keras_current_convnet_model_prediction(sample_ref_temp)
        conf_values = []
        image_values = []

        ####  THIS IS THE KEY BIT THAT WILL DIFFER FOR THE CONFIDENCE METHODS -
        ## DO WE TAKE THE LEAST-CONFIDENT, THE CASES WHERE THE MARGINAL CONF IS SMALLEST, OR WHERE THE ENTROPY OF CONFS IS HIGH?

        nd = []

        if method_of_conf_select == 'least_confidence':

            for n in range(len(node_data)):
                # print (n)
                # print ("node_data:", node_data[n])
                # print ("predictions:", node_data[n]['predictions'])
                print("confidence:", node_data[n]['confidence'])
                conf_values.append(node_data[n]['confidence'])

            conf_values = np.array(conf_values)
            indexes = conf_values.argsort()[:size]
            print("indexes:", indexes)
            print("Size of oo.point_select before: ", oo.point_select.shape)
            print("Size of oo.pca_distances_working_copy before: ", oo.pca_distances_working_copy.shape)
            for i in range(len(indexes)):
                obj = node_data[indexes[i]]
                nd.append(obj)
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                                          oo.pca_distances_working_copy,
                                                                                          obj['id'])
            print("Size of oo.point_select after: ", oo.point_select.shape)
            print("Size of oo.pca_distances_working_copy after: ", oo.pca_distances_working_copy.shape)

        elif method_of_conf_select == 'marginal_confidence':
            for n in range(len(node_data)):
                preds = node_data[n]['predictions']
                preds = np.sort(preds)
                print("Predicted values: ", preds)

                marginal = np.abs(preds[len(preds) - 1] - preds[len(preds) - 2])
                print("Marginal value: ", marginal)
                # print (n)
                # print ("node_data:", node_data[n])
                # print ("predictions:", node_data[n]['predictions'])
                # print ("confidence:", node_data[n]['confidence'])

                conf_values.append(marginal)

            conf_values = np.array(conf_values)
            indexes = conf_values.argsort()[:]
            indexes = np.flip(indexes, 0)
            indexes = indexes[:size]
            print("indexes:", indexes)
            print("Size of oo.point_select before: ", oo.point_select.shape)
            print("Size of oo.pca_distances_working_copy before: ", oo.pca_distances_working_copy.shape)
            for i in range(len(indexes)):
                obj = node_data[indexes[i]]
                nd.append(obj)
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                                          oo.pca_distances_working_copy,
                                                                                          obj['id'])
            print("Size of oo.point_select after: ", oo.point_select.shape)
            print("Size of oo.pca_distances_working_copy after: ", oo.pca_distances_working_copy.shape)

        elif method_of_conf_select == 'entropy_confidence':
            for n in range(len(node_data)):
                print(n)

                entropy = sp.stats.entropy(node_data[n]['predictions'])
                # print ("node_data:", node_data[n])
                print("predictions:", node_data[n]['predictions'])
                print("entropy:", entropy)
                # print ("confidence:", node_data[n]['confidence'])
                conf_values.append(entropy)

            conf_values = np.array(conf_values)
            indexes = conf_values.argsort()[:]
            indexes = np.flip(indexes, 0)
            indexes = indexes[:size]

            print("indexes:", indexes)
            print("Size of oo.point_select before: ", oo.point_select.shape)
            print("Size of oo.pca_distances_working_copy before: ", oo.pca_distances_working_copy.shape)
            for i in range(len(indexes)):
                obj = node_data[indexes[i]]
                nd.append(obj)
                oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                                          oo.pca_distances_working_copy,
                                                                                          obj['id'])
            print("Size of oo.point_select after: ", oo.point_select.shape)
            print("Size of oo.pca_distances_working_copy after: ", oo.pca_distances_working_copy.shape)

        return nd


def computeTsneAndPcaDistances():
    ### For each distance metric, we want a table that illustrates the following:
    ### X VALUE, Y VALUE, ACTUAL LABEL, PREDICTED LABEL, USER LABEL 
    X = np.array(oo.mnist.train.images)
    Y = np.array(oo.mnist.train.labels)
    actual_labels = Y.argmax(axis=1).reshape([X.shape[0], 1])
    user_labels = np.ones([actual_labels.shape[0], 1]) * -1
    user_confidences = np.zeros([actual_labels.shape[0], 1])
    loaded = False

    if oo.which_metric_to_use == 'pca':
        if os.path.exists(oo.pca_file):
            print("Loading PCA from file...")
            oo.pca_distances = np.loadtxt(oo.pca_file, delimiter=',')
            loaded = True
        else:
            print("Computing PCA... this can take some time...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(X)
            oo.pca_distances = pca.transform(X)
            this_filename = oo.pca_file
    elif oo.which_metric_to_use == 'umap':
        if os.path.exists(oo.umap_file):
            print("Loading UMAP from file...")
            oo.pca_distances = np.loadtxt(oo.umap_file, delimiter=',')
            loaded = True
        else:
            print("Computing UMAP... this can take some time...")
            import umap
            oo.pca_distances = umap.UMAP(n_neighbors=10, min_dist=0.001, metric='correlation').fit_transform(X)
            this_filename = oo.umap_file

    elif oo.which_metric_to_use == 'tsne':
        if os.path.exists(oo.tsne_file):
            print("Loading TSNE from file...")
            oo.pca_distances = np.loadtxt(oo.tsne_file, delimiter=',')
            loaded = True
        else:
            print("Computing TSNE... this can take some time...")
            from sklearn.manifold import TSNE
            oo.pca_distances = TSNE(n_components=2, verbose=1).fit_transform(X)
            this_filename = oo.tsne_file
            print("this_filename:", this_filename)

    elif oo.which_metric_to_use == 'pca_tsne':
        if os.path.exists(oo.pcatsne_file):
            print("Loading PCA-TSNE from file...")
            oo.pca_distances = np.loadtxt(oo.pcatsne_file, delimiter=',')
            loaded = True
        else:
            print("Computing PCA-TSNE... this can take some time...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            pca.fit(X)
            oo.pca_distances = pca.transform(X)
            from sklearn.manifold import TSNE
            oo.pca_distances = TSNE(n_components=2, verbose=1).fit_transform(oo.pca_distances)
            oo.pca_distances = np.hstack([oo.pca_distances, actual_labels])
            this_filename = oo.pcatsne_file
            print("this_filename:", this_filename)

    elif oo.which_metric_to_use == 'kmeans':
        if os.path.exists(oo.kmeans_centroid_file):
            print("Loading kMeans Centroid from file...")
            oo.pca_distances = np.loadtxt(oo.kmeans_centroid_file, delimiter=',')
            loaded = True
        else:
            print("Computing kMeans Centroid... this can take some time...")
            from sklearn.cluster import KMeans
            oo.pca_distances = KMeans(n_clusters=10, random_state=0).fit_transform(X)
            from sklearn.manifold import TSNE
            oo.pca_distances = TSNE(n_components=2, verbose=1).fit_transform(oo.pca_distances)
            oo.pca_distances = np.hstack([oo.pca_distances, actual_labels])
            this_filename = oo.kmeans_centroid_file
            print("this_filename:", this_filename)

    print("loaded", loaded)
    if not loaded:
        if oo.create_actual_label_column:
            oo.pca_distances = np.hstack([oo.pca_distances, actual_labels])

        if oo.create_predicted_label_column:
            from sklearn.cluster import KMeans
            clf = KMeans(n_clusters=oo.number_of_classes, random_state=42)
            clf.fit(X)
            labels = clf.labels_
            labels = labels.reshape([oo.pca_distances.shape[0], 1])
            oo.pca_distances = np.hstack([oo.pca_distances, labels])

        if oo.create_user_label_column:
            oo.pca_distances = np.hstack([oo.pca_distances, user_labels])

        if oo.create_user_confidence_column:
            oo.pca_distances = np.hstack([oo.pca_distances, user_confidences])

        oo.pca_distances[:, 0] = ((oo.pca_distances[:, 0] - np.min(oo.pca_distances[:, 0])) / (
                np.max(oo.pca_distances[:, 0]) - np.min(oo.pca_distances[:, 0])) * 2) - 1
        oo.pca_distances[:, 1] = ((oo.pca_distances[:, 1] - np.min(oo.pca_distances[:, 1])) / (
                np.max(oo.pca_distances[:, 1]) - np.min(oo.pca_distances[:, 1])) * 2) - 1
        print("Writing to file:", this_filename)
        np.savetxt(this_filename, oo.pca_distances, delimiter=',')

    index_count = np.array(range(oo.pca_distances.shape[0])).reshape(oo.pca_distances.shape[0], 1)
    oo.pca_distances = np.hstack([index_count, oo.pca_distances])
    oo.pca_distances_working_copy = np.copy(oo.pca_distances)

    oo.normal_distances = np.hstack([index_count, X])
    oo.normal_distances_working_copy = np.copy(oo.normal_distances)


def checkIfTrainChosen(sample):  # Seems to return None insted of alt sample?

    if (len(oo.total_train_sample_ref)) != (oo.total_train_pool):
        if sample in oo.total_train_sample_ref:
            alt_sample = rand.choice(os.listdir(oo.directory_for_train_images))
            return checkIfTestChosen(alt_sample)
        else:
            oo.total_train_sample_ref.append(sample)
            return sample
    else:
        print("Limit of reached at", i)


def oneHotEncoder(label):
    temp_train = np.zeros([oo.number_of_classes])
    temp_train[label] = 1
    temp_train = temp_train.reshape([1, oo.number_of_classes])
    print("temp_train:", temp_train)
    if (len(oo.y_train) > 0):
        oo.y_train = np.vstack([oo.y_train, temp_train])
    else:
        oo.y_train = temp_train
    print("y_train shape:", oo.y_train.shape)
    return temp_train


def keras_current_model_prediction(sample_batch):
    print("KERAS Current model prediction...")
    print("Sample batch length:", len(sample_batch))

    oo.temp_test = []

    # Recieves sample_ref array of mnist image names and converts:
    for i in range(len(sample_batch)):
        if oo.running_on_osx:
            img = Image.open(oo.directory_for_train_images + '/' + sample_batch[i]).convert('L')  # Greyscale
        else:
            img = Image.open(oo.directory_for_train_images + '\\' + sample_batch[i]).convert('L')  # Greyscale
        # Create an array with pixel values from image opened
        sample_vector = np.array(img).ravel() / 255
        # Appends to x_train
        if (len(oo.temp_test) > 0):
            oo.temp_test = np.vstack([oo.temp_test, sample_vector])
        else:
            oo.temp_test = sample_vector.reshape([1, sample_vector.shape[0]])

    model = model_from_json(open(oo.keras_model_filename[oo.model_to_load_from]).read())
    model.load_weights(oo.keras_weights_filename[oo.model_to_load_from])

    node_data = []

    # Assign predictions to y_train per models outcome:
    for i in range(len(sample_batch)):
        # print ("Predict for case ", i )

        prediction_result = model.predict((oo.temp_test[i, :]).reshape(1, 784))
        # print (prediction_result)

        data_entry = {}

        data_entry['id'] = sample_batch[i].split('_')[2]
        data_entry['id'] = int(data_entry['id'].split('.')[0])

        data_entry['label'] = int(np.argmax(prediction_result))
        data_entry['image'] = sample_batch[i]
        data_entry['confidence'] = float(np.max(prediction_result[0]))
        data_entry['predictions'] = prediction_result[0].tolist()

        node_data.append(data_entry)
        # oo.id_key.append(data_entry['id'])
        # oneHotEncoder( data_entry['label'] )

    return (node_data)


def keras_current_convnet_model_prediction(sample_batch):
    print("KERAS Current convnet model prediction...")
    print("Sample batch length:", len(sample_batch))

    oo.temp_test = []

    # Recieves sample_ref array of mnist image names and converts:
    for i in range(len(sample_batch)):
        if oo.running_on_osx:
            img = Image.open(oo.directory_for_train_images + '/' + sample_batch[i]).convert('L')  # Greyscale
        else:
            img = Image.open(oo.directory_for_train_images + '\\' + sample_batch[i]).convert('L')  # Greyscale
        # Create an array with pixel values from image opened
        sample_vector = np.array(img).ravel() / 255
        # Appends to x_train
        if (len(oo.temp_test) > 0):
            oo.temp_test = np.vstack([oo.temp_test, sample_vector])
        else:
            oo.temp_test = sample_vector.reshape([1, sample_vector.shape[0]])

        # np.append(x_train, sample_vector)
    # reshape x_train
    # x_train = x_train.reshape(int(len(x_train)/784), 784)

    print("Sample to run current model on ", oo.temp_test.shape)

    model = model_from_json(open(oo.keras_model_filename[oo.model_to_load_from]).read())
    model.load_weights(oo.keras_weights_filename[oo.model_to_load_from])

    node_data = []

    # Assign predictions to y_train per models outcome:
    for i in range(len(sample_batch)):
        print("Predict for case ", i)

        prediction_result = model.predict((oo.temp_test[i, :]).reshape(1, 28, 28, 1))
        print(prediction_result)
        data_entry = {}

        data_entry['id'] = sample_batch[i].split('_')[2]
        data_entry['id'] = int(data_entry['id'].split('.')[0])

        data_entry['label'] = int(np.argmax(prediction_result))
        data_entry['image'] = sample_batch[i]
        data_entry['confidence'] = float(np.max(prediction_result[0]))
        data_entry['predictions'] = prediction_result[0].tolist()
        node_data.append(data_entry)

        # oo.id_key.append(data_entry['id'])
        # oneHotEncoder( data_entry['label'] )

    return (node_data)


def appendConfidence(confidence_score):
    oo.confidence_scores.append(confidence_score)


def xTrainEncoder(image_ref):
    if oo.running_on_osx:
        img = Image.open(oo.directory_for_train_images + '/' + image_ref).convert('L')  # Greyscale
    else:
        img = Image.open(oo.directory_for_train_images + '\\' + image_ref).convert('L')  # Greyscale
    sample_vector = np.array(img).ravel() / 255
    if (len(oo.x_train) > 0):
        oo.x_train = np.vstack([oo.x_train, sample_vector])
    else:
        oo.x_train = sample_vector.reshape([1, sample_vector.shape[0]])
    print("X Train size: ", oo.x_train.shape)
    oo.batch_total = oo.x_train.shape[0]


def xTrainEncoderConfidence(image_ref, labels):
    if oo.running_on_osx:
        img = Image.open(oo.directory_for_train_images + '/' + image_ref).convert('L')  # Greyscale
    else:
        img = Image.open(oo.directory_for_train_images + '\\' + image_ref).convert('L')  # Greyscale
    sample_vector = np.array(img) / 255
    sample_vector = np.array(sample_vector).reshape(-1, 28, 28, 1)

    print("Lets generate new samples based on confidence")
    # print (sample_vector)
    # print (labels)

    counter = 0
    for X_gen, Y_gen in datagen.flow(sample_vector, labels, batch_size=1):
        sample_vector = X_gen.reshape([1, -1])
        if (len(oo.x_train) > 0):
            oo.x_train = np.vstack([oo.x_train, sample_vector])
        else:
            oo.x_train = sample_vector
        break

    print("X Train size: ", oo.x_train.shape)
    oo.batch_total = oo.x_train.shape[0]


def removeData(index):
    # overwrite both the image and label vector at index passed
    oo.x_train = np.delete(oo.x_train, index, axis=0)
    oo.y_train = np.delete(oo.y_train, index, axis=0)


def overwriteData(image_ref, label, index):
    # Open image from path passed in image_ref
    if oo.running_on_osx:
        img = Image.open(oo.directory_for_train_images + '/' + image_ref).convert('L')  # Greyscale
    else:
        img = Image.open(oo.directory_for_train_images + '\\' + image_ref).convert('L')  # Greyscale
    # Create an array with pixel values from image opened
    sample_vector = np.array(img).ravel() / 255
    print(index, oo.x_train.shape)
    x_train[index, :] = sample_vector
    # x_length = int(len(x_train))
    # x_train = x_train.reshape(x_length, 784)
    temp_train = np.eye(oo.number_of_classes)[label]
    oo.y_train[index, :] = temp_train
    print(index, oo.y_train.shape)
    # y_length = int(len(y_train))
    # y_train = y_train.reshape(y_length, 10)


def set_batch_total(n):
    oo.batch_total = oo.batch_total + int(n)
    return oo.batch_total


def generate_confusion_matrix(pred, actual):
    d = np.zeros([10, 10])
    print("Generate confusion matrix: ", len(pred))
    for i in range(len(pred)):
        row = actual[i];  #
        col = pred[i];  #

        print(row, col)
        d[row, col] = d[row, col] + 1  # + (1 / 10000)
    oo.conf_matrix = []
    for y in range(d.shape[0]):
        for x in range(d.shape[1]):
            entry = {}
            entry['x'] = x
            entry['y'] = y
            entry['value'] = d[y, x]
            oo.conf_matrix.append(entry)

    print(oo.conf_matrix)


@app.route("/get_conf_matrix")
def get_conf_matrix():
    if oo.train_step_log == 0:
        oo.conf_matrix = []
        for y in range(10):
            for x in range(10):
                entry = {}
                entry['x'] = x
                entry['y'] = y
                entry['value'] = 0
                oo.conf_matrix.append(entry)

    return json.dumps(oo.conf_matrix)


def keras_logreg(state, X_train, Y_train, file_set):
    iep = int(oo.epochs)
    ibi = int(oo.batches_in)
    ibs = int(oo.batch_size)
    its = int(oo.test_size)

    batch_size = ibs
    nb_classes = 10
    nb_epoch = iep
    input_dim = 784

    def build_logistic_model(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(output_dim, input_dim=input_dim))
        model.add(Activation('softmax'))
        return model

    print("Start of Keras logreg method")
    print("x_train", X_train.shape)
    print("y_train", Y_train.shape)

    X_test = (oo.mnist.test.images).reshape(-1, input_dim)
    Y_test = (oo.mnist.test.labels).reshape(-1, nb_classes)

    print("x_test", X_test.shape)
    print("y_test", Y_test.shape)

    if state > 0:
        model = model_from_json(open(oo.keras_model_filename[file_set]).read())
        model.load_weights(oo.keras_weights_filename[file_set])
    else:
        model = build_logistic_model(input_dim, nb_classes)

    model.summary()

    # compile the model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=1)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # outputs = [layer.output for layer in model.layers]
    # print (outputs)

    predict_vals = []
    actual_vals = []
    for i in range(Y_test.shape[0]):
        actual_vals.append(np.argmax(Y_test[i]))
        predict = model.predict(X_test[i].reshape(1, 784))

        predict_vals.append(np.argmax(predict[0]))
    generate_confusion_matrix(predict_vals, actual_vals)

    #### Simple way to get the output layer of network, and the predicted label
    print("Actual -- Label:", np.argmax(Y_test[0]), "Values:", Y_test[0])
    predict = model.predict(X_test[0].reshape(1, 784))
    print("Predict -- Label:", np.argmax(predict[0]), "Values:", predict, "Confidence:", np.max(predict[0]))

    # save model as json and yaml
    json_string = model.to_json()

    open(oo.keras_model_filename[file_set], 'w').write(json_string)
    # save the weights in h5 format
    model.save_weights(oo.keras_weights_filename[file_set])
    return score[1]


def keras_logreg_predicted_labels(state, input_data, output_labels, file_set):
    iep = int(oo.epochs)
    ibi = int(oo.batches_in)
    ibs = int(oo.batch_size)
    its = int(oo.test_size)

    batch_size = ibs
    nb_classes = 10
    nb_epoch = iep
    input_dim = 784

    def build_logistic_model(input_dim, output_dim):
        model = Sequential()
        model.add(Dense(output_dim, input_dim=input_dim))
        model.add(Activation('softmax'))
        return model

    print("Start of Keras logreg method")
    print("input_data", input_data.shape)
    print("output_labels", output_labels.shape)

    X_test = (oo.mnist.test.images).reshape(-1, input_dim)
    Y_test = (oo.mnist.test.labels).reshape(-1, nb_classes)

    print("x_test", X_test.shape)
    print("y_test", Y_test.shape)

    if state > 0:
        model = model_from_json(open(oo.keras_model_filename[file_set]).read())
        model.load_weights(oo.keras_weights_filename[file_set])
    else:
        model = build_logistic_model(input_dim, nb_classes)

    model.summary()

    # compile the model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(input_data, output_labels,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=1)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # outputs = [layer.output for layer in model.layers]
    # print (outputs)

    #### Simple way to get the output layer of network, and the predicted label
    print("Actual -- Label:", np.argmax(Y_test[0]), "Values:", Y_test[0])
    predict = model.predict(X_test[0].reshape(1, 784))
    print("Predict -- Label:", np.argmax(predict[0]), "Values:", predict, "Confidence:", np.max(predict[0]))

    # save model as json and yaml
    json_string = model.to_json()
    open(oo.keras_model_filename[file_set], 'w').write(json_string)
    # save the weights in h5 format
    model.save_weights(oo.keras_weights_filename[file_set])
    return score[1]


def keras_convnet(state, X_train, Y_train, file_set):
    iep = int(oo.epochs)
    ibi = int(oo.batches_in)
    ibs = int(oo.batch_size)
    its = int(oo.test_size)

    batch_size = ibs
    nb_classes = oo.number_of_classes
    nb_epoch = iep

    def build_convnet_model(input_dim, output_dim):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_dim))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        return model

    img_rows = 28
    img_cols = 28
    input_dim_convnet = (img_rows, img_cols, 1)

    print("Start of Keras convnet method")
    print("x_train", X_train.shape)
    print("y_train", Y_train.shape)

    X_test = (oo.mnist.test.images).reshape((oo.mnist.test.images).shape[0], img_rows, img_cols, 1)
    Y_test = (oo.mnist.test.labels).reshape(-1, nb_classes)

    print("x_test", X_test.shape)
    print("y_test", Y_test.shape)

    if state > 0:
        model = model_from_json(open(oo.keras_model_filename[file_set]).read())
        model.load_weights(oo.keras_weights_filename[file_set])
    else:
        model = build_convnet_model(input_dim_convnet, nb_classes)

    model.summary()

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=1)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    predict_vals = []
    actual_vals = []
    for i in range(Y_test.shape[0]):
        actual_vals.append(np.argmax(Y_test[i]))
        predict = model.predict(X_test[i].reshape(-1, img_rows, img_cols, 1))

        predict_vals.append(np.argmax(predict[0]))
    generate_confusion_matrix(predict_vals, actual_vals)

    #### Simple way to get the output layer of network, and the predicted label
    print("Actual -- Label:", np.argmax(Y_test[0]), "Values:", Y_test[0])
    predict = model.predict(X_test[0].reshape(-1, img_rows, img_cols, 1))
    print("Predict -- Label:", np.argmax(predict[0]), "Values:", predict, "Confidence:", np.max(predict[0]))

    # save model as json and yaml
    json_string = model.to_json()
    open(oo.keras_model_filename[file_set], 'w').write(json_string)
    # save the weights in h5 format
    model.save_weights(oo.keras_weights_filename[file_set])
    return score[1]


def keras_convnet_predicted_labels(state, input_data, output_labels, file_set):
    iep = int(oo.epochs)
    ibi = int(oo.batches_in)
    ibs = int(oo.batch_size)
    its = int(oo.test_size)

    batch_size = ibs
    nb_classes = oo.number_of_classes
    nb_epoch = iep

    def build_convnet_model(input_dim, output_dim):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_dim))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_dim, activation='softmax'))
        return model

    img_rows = 28
    img_cols = 28
    input_dim_convnet = (img_rows, img_cols, 1)

    X_train = input_data.reshape(-1, img_rows, img_cols, 1)
    Y_train = output_labels.reshape(-1, nb_classes)

    print("Start of Keras convnet method")
    print("x_train", X_train.shape)
    print("y_train", Y_train.shape)

    X_test = (oo.mnist.test.images).reshape((oo.mnist.test.images).shape[0], img_rows, img_cols, 1)
    Y_test = (oo.mnist.test.labels).reshape(-1, nb_classes)

    print("x_test", X_test.shape)
    print("y_test", Y_test.shape)

    if state > 0:
        model = model_from_json(open(oo.keras_model_filename[file_set]).read())
        model.load_weights(oo.keras_weights_filename[file_set])
    else:
        model = build_convnet_model(input_dim_convnet, nb_classes)

    model.summary()

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=nb_epoch,
                        verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=1)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # outputs = [layer.output for layer in model.layers]
    # print (outputs)

    #### Simple way to get the output layer of network, and the predicted label
    print("Actual -- Label:", np.argmax(Y_test[0]), "Values:", Y_test[0])
    predict = model.predict(X_test[0].reshape(-1, img_rows, img_cols, 1))
    print("Predict -- Label:", np.argmax(predict[0]), "Values:", predict, "Confidence:", np.max(predict[0]))

    # save model as json and yaml
    json_string = model.to_json()
    open(oo.keras_model_filename[file_set], 'w').write(json_string)
    # save the weights in h5 format
    model.save_weights(oo.keras_weights_filename[file_set])
    return score[1]


@app.route("/predict_pool_labels")
def predict_pool_labels():
    print("predict_pool_labels")
    # print ("Known user labels")
    known_user_labels = oo.pca_distances[:, 5]
    # print (known_user_labels)
    mask = (known_user_labels > -1)

    set_of_known_points = oo.pca_distances[mask, :]

    # print ("set_of_known_points")
    # print (set_of_known_points)

    for i in range(oo.pca_distances.shape[0]):
        x = oo.pca_distances[i, 1]
        y = oo.pca_distances[i, 2]
        min_d = 5000
        min_j = -1
        for j in range(set_of_known_points.shape[0]):
            d = np.sqrt((x - set_of_known_points[j, 1]) ** 2 + (y - set_of_known_points[j, 2]) ** 2)
            if d < min_d:
                min_d = d
                min_j = set_of_known_points[j, 5]
        oo.pca_distances[i, 4] = min_j

    output = {}
    output['some_data'] = oo.pca_distances[:, 4].tolist()
    return json.dumps(output)


@app.route("/set_batch_size")
def set_batch_size():
    oo.batch_size = request.args.get('batch_size_input')
    print("batch size set to ", oo.batch_size)
    return oo.batch_size


@app.route("/set_epoch")
def set_epoch():
    oo.epochs = request.args.get('epoch_num')
    return oo.epochs


@app.route("/set_batches_in")
def set_batches_in():
    oo.batches_in = request.args.get('batches_in')
    return oo.batches_in


@app.route("/get_oo.batch_total")
def get_batch_total():
    output = {}
    output['oo.batch_total'] = oo.batch_total
    return (json.dumps(output))


@app.route("/get_samples")
def get_samples():
    object_list = []
    if oo.sample_selection_method == 'random':
        object_list = getRandomTrainBatch(int(oo.batch_size))
    elif oo.sample_selection_method == 'distance':
        object_list = getDistanceTrainBatch(int(oo.batch_size))
    elif oo.sample_selection_method == 'confidence':
        object_list = getConfidenceRandomTrainBatch(int(oo.batch_size), method_of_conf_select='least_confidence')
    elif oo.sample_selection_method == 'confidence_distance':
        object_list = getConfidenceDistanceTrainBatch(int(oo.batch_size), method_of_conf_select='least_confidence')
    elif oo.sample_selection_method == 'confidence_marginal_random':
        object_list = getConfidenceRandomTrainBatch(int(oo.batch_size), method_of_conf_select='marginal_confidence')
    elif oo.sample_selection_method == 'confidence_marginal_distance':
        object_list = getConfidenceDistanceTrainBatch(int(oo.batch_size), method_of_conf_select='marginal_confidence')
    elif oo.sample_selection_method == 'confidence_entropy_random':
        object_list = getConfidenceRandomTrainBatch(int(oo.batch_size), method_of_conf_select='entropy_confidence')
    elif oo.sample_selection_method == 'confidence_entropy_distance':
        object_list = getConfidenceDistanceTrainBatch(int(oo.batch_size), method_of_conf_select='entropy_confidence')
    print("object list")
    for objs in object_list:
        print(objs)
        if isinstance(objs, str):
            oo.all_selected_filenames.append(objs)
        elif isinstance(objs, dict):
            print("add to all_selected_filenames dict")
            oo.all_selected_filenames.append(objs['image'])
    print("Number of objects in list:", len(object_list))
    print("Length of oo.all_selected_filenames: ", len(oo.all_selected_filenames))
    print("Unique length of oo.all_selected_filenames: ", len(list(set(oo.all_selected_filenames))))
    return (json.dumps(object_list))


@app.route("/get_train_step_log")
def get_train_step_log():
    output = {}
    output['train_step_log'] = oo.train_step_log
    return (json.dumps(output))


@app.route('/update_label')
def update_label():
    # If node already has a label, and is moves to produce a new label,
    # the old label needs to be replaced rather than appended again.

    node_id = str(request.args.get('id'))
    node_label = str(request.args.get('label'))
    node_image = str(request.args.get('image'))

    print("node_image:", node_image)
    xpos = float(request.args.get('xpos'))
    ypos = float(request.args.get('ypos'))
    user_confidence = float(request.args.get('user_confidence'))
    user_labelled = str(request.args.get('user_labelled'))

    new_label = int(request.args.get('new_label'))
    if (new_label == -1):
        new_label = "?"

    # batch_labels = np.array([[]]).astype(int)
    output = {}

    file_id = int(node_image.split(".")[0].split("_")[2])

    # output['scatter_index'] = oo.pca_distances[file_id, 0]
    output['scatter_x'] = oo.pca_distances[file_id, 1]
    output['scatter_y'] = oo.pca_distances[file_id, 2]
    output['label'] = new_label

    output['total_samples'] = str(oo.x_train.shape[0])

    # Only if a label is produced will the code below run:
    # Check to see if node as been labeled before:

    if node_label != "?" and new_label == "?":
        oo.pca_distances[file_id, 5] = -1
        oo.pca_distances[file_id, 6] = oo.pca_distances[file_id, 6] + 1
        print(file_id)
        print(oo.pca_distances[file_id, :])

        if node_id in oo.id_key:
            index = oo.id_key.index(node_id)
            # print ("Remove entry from x_train matrix")
            # removeData(index)

        output['total_samples'] = str(oo.x_train.shape[0])
        return json.dumps(output)

    ### COMMENTED OUT FOR THE MOMENT - THIS DEALS WITH WHEN AN EXISTING INSTANCE
    ### IS RELABELLED - HOWEVER LET'S JUST ASSUME FOR NOW WE POSITION INSTANCE ONLY ONCE

    # overwriteData(node_image, new_label, index)

    elif new_label != "?":
        if node_id in oo.id_key:
            print("node_id in oo.id_key - likely that just been classed by machine???")
            index = oo.id_key.index(node_id)
        else:
            oo.id_key.append(node_id)
            # if oo.use_confidence:
            #    generate_samples_based_on_confidence(new_label, node_image, user_confidence)
            #    print (" * Size of x_train: ", oo.x_train.shape)
            #    print (" * Size of y_train: ", oo.y_train.shape)
            # else:

            oneHotEncoder(new_label)
            xTrainEncoder(node_image)
            appendConfidence(user_confidence)

            print("-- Update to user provided labels ---")
            # actual label: oo.pca_distances[file_id, 3]
            # oo.pca_distances[file_id, 4] prediction by system
            oo.pca_distances[file_id, 5] = int(new_label)
            oo.pca_distances[file_id, 6] = user_confidence
            print("file_id:", file_id)
            print("oo.pca_distances[file_id,:]", oo.pca_distances[file_id, :])

            print("==>image id {}, label {}, confidence {} ".format(file_id, int(new_label), user_confidence))
            # create confidence file with header row if it does not already exist
            # output confidence details to file
            if (os.path.isfile('confidences.txt') == False):
                with open('confidences.txt', 'w') as thefile:
                    print('ImageId,Label,Confidence', file=thefile)
                    print("{}, {}, {} ".format(file_id, int(new_label), user_confidence), file=thefile)
            else:
                with open('confidences.txt', 'a') as thefile:
                    print("{}, {}, {} ".format(file_id, int(new_label), user_confidence), file=thefile)

        output['total_samples'] = str(oo.x_train.shape[0])
        return json.dumps(output)


@app.route("/add_image_to_labelling_pool")
def add_image_to_labelling_pool():
    my_index = int(request.args.get('my_index'))
    print("add_image_to_labelling_pool", my_index)
    print("Add image", my_index, "to the labelling pool")
    oo.point_select, oo.pca_distances_working_copy, this_index = select_point(oo.point_select,
                                                                              oo.pca_distances_working_copy, my_index)
    output = {}

    sample = 'MNIST_image_' + str(int(my_index)) + '.png'
    sample_ref = []
    sample_ref.append(sample)

    ### one thing currently missing - if user selects from sample pool and classifier exists
    ### then should we return the predicted label and position with this?
    if oo.train_step_log > 0:
        if oo.use_convnet == 'keraslogreg':
            node_data = keras_current_model_prediction(sample_ref)
            output['node_data'] = node_data
        elif oo.use_convnet == 'kerasconvnet':
            node_data = keras_current_convnet_model_prediction(sample_ref)
            output['node_data'] = node_data

    return json.dumps(output)


@app.route("/get_image_from_scatter_position")
def get_image_from_scatter_position():
    output = {}
    cx = float(request.args.get('cx'))
    cy = float(request.args.get('cy'))
    mouse = np.array([cx, cy])
    points = oo.pca_distances[:, 0:2]
    c = np.abs(points - mouse)
    d = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
    print("Position: ", cx, cy)
    print(mouse)
    print(points.shape)
    print(c)
    print(d)
    index = np.argmin(d)
    print("Index:", index)
    output['index'] = int(index)
    return json.dumps(output)


@app.route("/prepare_bar_data")
def prepare_bar_data():
    output = {}
    output['batchsize'] = int(oo.batch_size)
    output['testsize'] = int(oo.test_size)
    output['poolsize'] = int(oo.total_pool_size)
    return json.dumps(output)


@app.route("/prepare_line_graph_data")
def prepare_line_graph_data():
    output = {}
    output['samples'] = int(oo.batch_total)
    output['accuracy'] = oo.current_accuracy
    return json.dumps(output)


@app.route("/prepare_scatter_plot_data")
def prepare_scatter_plot_data():
    output = oo.pca_distances[:, 1:]
    print("Prepare scatterplot:", output.shape)
    output = output.tolist()
    return json.dumps(output)


@app.route("/train_model_using_all_methods")
def train_model_using_all_methods():
    print("Train the model using four possible methods")
    print("   1. Single instance user labels")
    print("   2. Inferred labels for full training set")
    print("   3. Data Augmentation using ImageDataGenerator")
    print("   4. Data Augmentation using User Confidence")

    user_label_counter = request.args.get('user_label_counter')

    output = {}

    sample_size = 0
    si = 0
    il = 0
    da = 0
    ca = 0

    if (oo.perform_single_instance):
        oo.use_sample_generator = 'none'
        first = train_model_using_user_labels()
        output['single_instance'] = json.loads(first)
        sample_size = output['single_instance']['samples']
        si = output['single_instance']['accuracy']

    if (oo.perform_inferred_label):
        predict_pool_labels()
        second = train_model_using_predicted_labels()
        output['inferred_label'] = json.loads(second)
        sample_size = output['inferred_label']['samples']
        il = output['inferred_label']['accuracy']

    if (oo.perform_data_augmentation):
        oo.use_sample_generator = 'standard'
        third = train_model_using_user_labels()
        output['data_augmentation'] = json.loads(third)
        sample_size = output['data_augmentation']['samples']
        da = output['data_augmentation']['accuracy']

    if (oo.perform_confidence_augmentation):
        oo.use_sample_generator = 'confidence'
        fourth = train_model_using_user_labels()
        output['confidence_augmentation'] = json.loads(fourth)
        sample_size = output['confidence_augmentation']['samples']
        ca = output['confidence_augmentation']['accuracy']

    oo.train_step_log = oo.train_step_log + 1

    print("Just to check we have not modified x_train and y_train")
    print("  x_train shape:", oo.x_train.shape)
    print("  y_train shape:", oo.y_train.shape)

    output['sample_size'] = oo.batch_size

    with open(oo.results_file, 'a') as the_file:
        the_file.write(str(sample_size) + "," + str(si) + "," + str(il) + "," + str(da) + "," + str(
            ca) + "," + user_label_counter + "\n")
    return json.dumps(output)


def perform_standard_generator():
    shape_of_x_train = np.array(oo.x_train).shape
    print("shape_of_x_train", shape_of_x_train)
    loop_count = oo.x_train.shape[0]

    batch_size = 32
    print("Batch size:", batch_size)

    X_train = np.array(oo.x_train).reshape(-1, 28, 28, 1)
    Y_train = np.array(oo.y_train).reshape(-1, 10)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)

    batches = 0
    newX = []
    newY = []

    for X_gen, Y_gen in datagen.flow(X_train, Y_train, batch_size=batch_size):
        print(X_gen.shape, Y_gen.shape)
        if batches == 0:
            newX = X_gen
            newY = Y_gen
        else:
            print("newX", newX.shape, "X_gen", X_gen.shape)
            newX = np.vstack([newX, X_gen])
            newY = np.vstack([newY, Y_gen])
        batches += 1
        if batches == loop_count:
            break

    print("newX shape:", newX.shape)
    print("newY shape:", newY.shape)
    return newX, newY


def perform_confidence_generator():
    shape_of_x_train = np.array(oo.x_train).shape
    print("shape_of_x_train", shape_of_x_train)
    batch_size = 32

    loop_count = oo.x_train.shape[0]

    Xconf = []
    Yconf = []

    print("len(oo.confidence_scores): ", len(oo.confidence_scores))
    print("oo.x_train.shape:", oo.x_train.shape)

    for i in range(len(oo.confidence_scores)):

        conf_score = int(oo.confidence_scores[i] * oo.confidence_weight)
        print("i:", i, "conf score:", conf_score)

        Xreps = np.tile(oo.x_train[i, :], (conf_score, 1))
        Yreps = np.tile(oo.y_train[i, :], (conf_score, 1))

        print("Xreps.shape:", Xreps.shape)

        if i == 0:
            Xconf = Xreps
            Yconf = Yreps
        else:
            Xconf = np.vstack([Xconf, Xreps])
            Yconf = np.vstack([Yconf, Yreps])

    print("Xconf shape:", Xconf.shape)
    print("Yconf shape:", Yconf.shape)

    X_train = np.array(Xconf).reshape(-1, 28, 28, 1)
    Y_train = np.array(Yconf).reshape(-1, 10)
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    batches = 0
    newX = []
    newY = []
    for X_gen, Y_gen in datagen.flow(X_train, Y_train, batch_size=batch_size):
        print(X_gen.shape, Y_gen.shape)
        if batches == 0:
            newX = X_gen
            newY = Y_gen
        else:
            print("newX", newX.shape, "X_gen", X_gen.shape)
            newX = np.vstack([newX, X_gen])
            newY = np.vstack([newY, Y_gen])
        batches += 1
        if batches == loop_count:
            break
    print("newX shape:", newX.shape)
    print("newY shape:", newY.shape)
    return newX, newY


@app.route("/train_model_using_user_labels")
def train_model_using_user_labels():
    print("Train the model using only the labels provided by the user")
    print("   i.e. a subset of the original training dataset")
    output = {}

    if oo.use_convnet == 'keraslogreg':
        if oo.use_sample_generator == 'standard':
            print("Use logreg with standard generator")
            newX, newY = perform_standard_generator()
            X_train = newX.reshape(-1, 784)
            Y_train = newY.reshape(-1, 10)
            print("X_train shape:", X_train.shape)
            print("Y_train shape:", Y_train.shape)
            accuracy = keras_logreg(oo.train_step_log, X_train, Y_train, file_set=2)

        elif oo.use_sample_generator == 'confidence':
            print("Use logres with confidence generator")
            newX, newY = perform_confidence_generator()

            X_train = newX.reshape(-1, 784)
            Y_train = newY.reshape(-1, 10)
            print("X_train shape:", X_train.shape)
            print("Y_train shape:", Y_train.shape)
            accuracy = keras_logreg(oo.train_step_log, X_train, Y_train, file_set=3)

        elif oo.use_sample_generator == 'none':
            X_train = np.array(oo.x_train).reshape(-1, 784)
            Y_train = np.array(oo.y_train).reshape(-1, 10)
            accuracy = keras_logreg(oo.train_step_log, X_train, Y_train, file_set=0)
            # oo.train_step_log = oo.train_step_log + 1
    elif oo.use_convnet == 'kerasconvnet':
        if oo.use_sample_generator == 'standard':
            print("Use convnet with standard generator")

            newX, newY = perform_standard_generator()

            X_train = newX.reshape(-1, 28, 28, 1)
            Y_train = newY.reshape(-1, 10)
            print("X_train shape:", X_train.shape)
            print("Y_train shape:", Y_train.shape)
            accuracy = keras_convnet(oo.train_step_log, X_train, Y_train, file_set=2)

        elif oo.use_sample_generator == 'confidence':
            print("Use convnet with confidence generator")

            newX, newY = perform_confidence_generator()

            X_train = newX.reshape(-1, 28, 28, 1)
            Y_train = newY.reshape(-1, 10)
            print("X_train shape:", X_train.shape)
            print("Y_train shape:", Y_train.shape)
            accuracy = keras_convnet(oo.train_step_log, X_train, Y_train, file_set=3)

        elif oo.use_sample_generator == 'none':
            X_train = np.array(oo.x_train).reshape(-1, 28, 28, 1)
            Y_train = np.array(oo.y_train).reshape(-1, 10)
            accuracy = keras_convnet(oo.train_step_log, X_train, Y_train, file_set=0)
            # oo.train_step_log = oo.train_step_log + 1

    output['samples'] = len(oo.x_train)
    output['accuracy'] = int(accuracy * 100)
    return json.dumps(output)


@app.route("/train_model_using_predicted_labels")
def train_model_using_predicted_labels():
    print("Train the model using the predicted labels for the complete training dataset")

    X = np.array(oo.mnist.train.images)
    Y = np.array(oo.mnist.train.labels)

    print("Y size:", Y.shape)

    labels_Y = []

    for i in range(oo.pca_distances.shape[0]):
        row = np.zeros([1, 10])
        pos = int(oo.pca_distances[i, 4])
        row[0, pos] = 1

        if i == 0:
            labels_Y = row
        else:
            labels_Y = np.vstack([labels_Y, row])
        # oo.pca_distances[i,4]

    print(labels_Y.shape)
    print(labels_Y[0])
    Y = labels_Y
    print("Y size:", Y.shape)

    if oo.use_convnet == 'keraslogreg':
        accuracy = keras_logreg_predicted_labels(oo.train_step_log, X, Y, file_set=1)
        # oo.train_step_log = oo.train_step_log + 1
    elif oo.use_convnet == 'kerasconvnet':
        accuracy = keras_convnet_predicted_labels(oo.train_step_log, X, Y, file_set=1)
        # oo.train_step_log = oo.train_step_log + 1

    # Output accuracy of current model
    # Cast output to int so truncated and JSON serialisable
    output = {}
    output['samples'] = int(X.shape[0])
    output['accuracy'] = int(accuracy * 100)
    return json.dumps(output)


@app.route("/ml_select")
def ml_select():
    value = request.args.get('ml_select')
    oo.use_convnet = value
    print("oo.use_convnet", oo.use_convnet)
    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/distance_select")
def distance_select():
    value = request.args.get('distance_select')
    if value == 'orig_dim':
        oo.sample_selection_method = 'distance'
        oo.use_dimensional_reduction_mode = False
    elif value == 'random':
        oo.sample_selection_method = 'random'
    elif value == 'reduce_dim':
        oo.sample_selection_method = 'distance'
        oo.use_dimensional_reduction_mode = True
    elif value == 'confidence':
        oo.sample_selection_method = 'confidence'
    elif value == 'confidence_distance':
        oo.sample_selection_method = 'confidence_distance'
    elif value == 'confidence_marginal_distance':
        oo.sample_selection_method = 'confidence_marginal_distance'
    elif value == 'confidence_marginal_random':
        oo.sample_selection_method = 'confidence_marginal_random'
    elif value == 'confidence_entropy_distance':
        oo.sample_selection_method = 'confidence_entropy_distance'
    elif value == 'confidence_entropy_random':
        oo.sample_selection_method = 'confidence_entropy_random'

    print("oo.use_dimensional_reduction_mode", oo.use_dimensional_reduction_mode)
    print("oo.sample_selection_method", oo.sample_selection_method)
    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/dimension_select")
def dimension_select():
    value = request.args.get('dimension_select')
    if value == 'umap':
        oo.which_metric_to_use = 'umap'
        computeTsneAndPcaDistances()
    elif value == 'pca':
        oo.which_metric_to_use = 'pca'
        computeTsneAndPcaDistances()
    elif value == 'tsne':
        oo.which_metric_to_use = 'tsne'
        computeTsneAndPcaDistances()
    elif value == 'kmeans':
        oo.which_metric_to_use = 'kmeans'
        computeTsneAndPcaDistances()
    elif value == 'pca_tsne':
        oo.which_metric_to_use = 'pca_tsne'
        computeTsneAndPcaDistances()
    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/generator_select")
def generator_select():
    value = request.args.get('generator_select')
    if value == 'true':
        oo.use_sample_generator = True
    elif value == 'false':
        oo.use_sample_generator = False
    print("Use sample generator:", oo.use_sample_generator)
    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/confidence_select")
def confidence_select():
    value = request.args.get('confidence_select')
    if value == 'true':
        oo.use_confidence = True
    elif value == 'false':
        oo.use_confidence = False
    print("Use confidence:", oo.use_confidence)
    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/method_select")
def method_select():
    value1 = request.args.get('value1')
    value2 = request.args.get('value2')
    value3 = request.args.get('value3')
    value4 = request.args.get('value4')

    oo.perform_single_instance = False
    oo.perform_inferred_label = False
    oo.perform_data_augmentation = False
    oo.perform_confidence_augmentation = False

    if value1 == 'true':
        oo.perform_single_instance = True
    if value2 == 'true':
        oo.perform_inferred_label = True
    if value3 == 'true':
        oo.perform_data_augmentation = True
    if value4 == 'true':
        oo.perform_confidence_augmentation = True

    print("oo.perform_single_instance:", oo.perform_single_instance)
    print("oo.perform_inferred_label:", oo.perform_inferred_label)
    print("oo.perform_data_augmentation:", oo.perform_data_augmentation)
    print("oo.perform_confidence_augmentation:", oo.perform_confidence_augmentation)

    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/sample_selection_method")
def sample_selection_method():
    value = request.args.get('method')
    if value == 'random':
        oo.sample_selection_method = 'random'
    elif value == 'distance':
        oo.sample_selection_method = 'distance'
    print("Sample selection method:", oo.sample_selection_method)
    output = {'result': 'done'}
    return json.dumps(output)


@app.route('/machine_predict_select')
def machine_predict_select():
    value = request.args.get('machine_predict_select')
    oo.model_to_load_from = int(value)

    print("machine_predict_select set to: ", value)
    print(oo.model_to_load_from)
    print(oo.keras_model_filename[oo.model_to_load_from])
    print(oo.keras_weights_filename[oo.model_to_load_from])

    output = {'result': 'done', 'value': value}
    return json.dumps(output)


@app.route("/sample_pool")
def sample_pool():
    return render_template("sample_pool.html")


@app.route("/sample_predict")
def sample_predict():
    return render_template("sample_predict.html")


@app.route("/confusion_matrix")
def confusion_matrix():
    return render_template("confusion_matrix.html")


@app.route("/set_line_chart_file")
def set_line_chart_file():
    value = request.args.get('filename')
    oo.file_name_for_line_chart_to_display = value
    output = {'result': 'done'}
    return json.dumps(output)


@app.route("/get_data_to_populate_line_chart")
def get_data_to_populate_line_chart():
    filename = './results/' + oo.file_name_for_line_chart_to_display + '.csv'
    file_data = np.loadtxt(filename, skiprows=1, delimiter=',')
    ### now let's send this to the line chart
    data_in_correct_format = []

    if file_data.shape[1] == 5:
        for i in range(file_data.shape[0]):
            samples = file_data[i, 0]
            single = file_data[i, 1]
            inferred = file_data[i, 2]
            augment = file_data[i, 3]
            conf = file_data[i, 4]
            entry = {}
            entry['samples'] = samples
            entry['accuracies'] = {}
            entry['accuracies']['single_instance'] = single
            entry['accuracies']['inferred_label'] = inferred
            entry['accuracies']['data_augmentation'] = augment
            entry['accuracies']['confidence_augmentation'] = conf
            data_in_correct_format.append(entry)
        output = data_in_correct_format
        return json.dumps(output)
    elif file_data.shape[1] == 6:
        for i in range(file_data.shape[0]):
            samples = file_data[i, 0]
            single = file_data[i, 1]
            inferred = file_data[i, 2]
            augment = file_data[i, 3]
            conf = file_data[i, 4]
            user_labels = file_data[i, 5]
            entry = {}
            entry['samples'] = samples
            entry['accuracies'] = {}
            entry['accuracies']['single_instance'] = single
            entry['accuracies']['inferred_label'] = inferred
            entry['accuracies']['data_augmentation'] = augment
            entry['accuracies']['confidence_augmentation'] = conf
            entry['accuracies']['user_label_counter'] = user_labels
            data_in_correct_format.append(entry)
        output = data_in_correct_format
        return json.dumps(output)


@app.route("/test_accuracy_chart")
def test_accuracy_chart():
    return render_template("test_accuracy_chart.html")


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown')
def shutdown():
    shutdown_server()
    output = 'Server shutting down...'
    return output


@app.route("/")
def index():
    global oo
    print("Previous batch size:", oo.getBatchTotal())
    print("x_train:", oo.x_train.shape)
    import activate_config
    reload(activate_config)
    from activate_config import Options
    oo = Options()
    print("Current batch size:", oo.getBatchTotal())
    print("x_train:", oo.x_train.shape)
    computeTsneAndPcaDistances()
    print("id_key: ", oo.id_key)
    oo.results_file = "results_" + str(datetime.datetime.now())
    oo.results_file = oo.results_file.replace(" ", "_")
    oo.results_file = oo.results_file.replace(":", "_")
    oo.results_file = oo.results_file.replace(".", "_")
    oo.results_file = "./results/" + oo.results_file + ".csv"
    with open(oo.results_file, 'w') as the_file:
        the_file.write('samples,single,inferred,imageaugment,confaugment,userlabels\n')
    # return render_template("v4.html")
    return render_template("v5.html")


if __name__ == "__main__":
    import sys

    print("args length ", len(sys.argv), sys.argv)
    port_number = 7234  # default
    if len(sys.argv) < 2:
        print("Usage: ")
        print("  python activate_app.py --port 7234")
        print(" --- ")
        print("  Will continue on default port 7234")
    else:
        if (sys.argv[1]) == '--port' and len(sys.argv) == 3:
            print("Do we get here?")
            port_number = int(sys.argv[2])

    print("Running on port number: ", port_number)
    app.run(host='0.0.0.0', port=port_number, debug=True)
