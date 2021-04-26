# Activate config file

import numpy as np

from tensorflow import keras
from pathlib import Path


def get_label_array(argument):  # Labels stored as digits 0-9 in dataset
    switcher = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    return switcher.get(argument)


class Options:
    use_convnet = 'keraslogreg'
    use_dimensional_reduction_mode = True
    which_metric_to_use = 'umap'  # could be pca, umap, tsne, pca_tsne
    create_actual_label_column = True
    create_predicted_label_column = True
    create_user_label_column = True
    create_user_confidence_column = True
    n_clusters = 10
    sample_selection_method = 'random'

    results_file = ''

    file_name_for_line_chart_to_display = 'results_2018-03-25_14_24_02_290656'

    perform_single_instance = True
    perform_inferred_label = True
    perform_data_augmentation = True
    perform_confidence_augmentation = True

    conf_matrix = []

    batch_size = 10
    batch_total = 0
    test_size = 0

    epochs = 0
    batches_in = 0
    current_accuracy = 0
    train_step_log = 0
    logreg_saver = None
    id_key = []
    total_test_sample_ref = []
    total_train_sample_ref = []

    temp_test = []
    confidence_scores = []
    confidence_weight = 100

    all_selected_filenames = []

    # keras_model_filename = 'keras_models/model.json'
    # keras_weights_filename = 'keras_models/model_weights.h5'

    keras_dir = 'keras_models/'
    keras_model_filename = [keras_dir + 'single.json', keras_dir + 'inferred.json', keras_dir + 'imgaugment.json',
                            keras_dir + 'confidence.json', ]
    keras_weights_filename = [keras_dir + 'single_weights.h5', keras_dir + 'inferred_weights.h5',
                              keras_dir + 'imgaugment_weights.h5', keras_dir + 'confidence_weights.h5', ]

    model_to_load_from = 3

    W = None
    b = None
    tsne_distances = []
    pca_distances = []
    tsne_distances_working_copy = []
    pca_distances_working_copy = []
    normal_distances = []
    normal_distances_working_copy = []
    point_select = []
    use_sample_generator = False
    use_confidence = False

    # Load data set
    # (training image, training lables), (test images, test labels)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    # Normalise vals
    train_images, test_images = train_images / 255.0, test_images / 255.0

    total_test_pool = len(test_images)
    total_train_pool = len(train_images)

    number_of_classes = 10

    # Reshape
    train_images = np.delete(train_images, np.s_[total_train_pool:], axis=0)
    # Dynamic reshape size
    temp = int(train_images.size / total_train_pool)

    train_images = train_images.reshape(total_train_pool, temp)
    test_images = test_images.reshape(total_test_pool, temp)

    # Convert new style labels back to binary array of [0,9]
    temp = []
    train_labels = np.delete(train_labels, np.s_[total_train_pool:], axis=0)
    for value in train_labels:
        temp.append(get_label_array(int(value)))

    train_labels = np.array(temp)

    del temp[:]
    for value in test_labels:
        temp.append(get_label_array(int(value)))

    test_labels = np.array(temp)

    directory_for_test_images = Path("static/images/cifar10_keras/test")
    directory_for_train_images = Path("static/images/cifar10_keras/train")
    directory_for_trained_model = Path("logs/trained_logreg_model.ckpt")

    tsne_file = Path("distance_data/tsne_distances.csv")
    pca_file = Path("distance_data/pca_distances.csv")
    umap_file = Path("distance_data/umap_distances.csv")
    pcatsne_file = Path("distance_data/pcatsne_distances.csv")
    kmeans_centroid_file = Path("distance_data/kmeans_centroid_distances.csv")

    eye_tracking_start_file = Path("start.txt")

    # Run eye tracker
    command = '../EyeTrackerCode/main.exe'.split()

    def __init__(self):
        print("__init__ called")
        print("use_convnet:", self.use_convnet)
        print("id_key:", self.id_key)

    def getBatchTotal(self):
        print("use_convnet:", self.use_convnet)
        return self.batch_total
