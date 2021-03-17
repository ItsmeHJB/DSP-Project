import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow import keras

from sys import platform


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
    sample_selection_method = 'distance'

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
    total_test_pool = 10000
    total_train_pool = 50000

    number_of_classes = 10

    point_select = []

    epochs = 0
    batches_in = 0
    current_accuracy = 0
    train_step_log = 0
    logreg_saver = None
    id_key = []
    total_test_sample_ref = []
    total_train_sample_ref = []
    train_images = np.zeros([0, 784])
    train_labels = np.zeros([0, 10])
    test_images = []  # np.zeros([1,784])
    test_labels = []  # np.zeros([1,10])

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

    if platform == "linux" or platform == "linux2":
        running_on_osx = True
    elif platform == "darwin":
        running_on_osx = True
    elif platform == "win32":
        running_on_osx = False

    # Load data set
    # (training image, training lables), (test images, test labels)
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    # Normalise vals
    train_images, test_images = train_images / 255.0, test_images / 255.0

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

    if running_on_osx:
        directory_for_test_images = "static/images/cifar10_keras/test"
        directory_for_train_images = "static/images/cifar10_keras/train"
        directory_for_trained_model = "logs/trained_logreg_model.ckpt"

        tsne_file = "./distance_data/tsne_distances.csv"
        pca_file = "./distance_data/pca_distances.csv"
        umap_file = "./distance_data/umap_distances.csv"
        pcatsne_file = "./distance_data/pcatsne_distances.csv"
        kmeans_centroid_file = "./distance_data/kmeans_centroid_distances.csv"
    else:
        # directory_for_test_images = "static\\images\\cifar10\\test"
        # directory_for_train_images = "static\\images\\cifar10\\train"
        directory_for_test_images = "static\\images\\cifar10_keras\\test"
        directory_for_train_images = "static\\images\\cifar10_keras\\train"
        directory_for_trained_model = "logs\\trained_logreg_model.ckpt"

        tsne_file = "distance_data\\tsne_distances.csv"
        pca_file = "distance_data\\pca_distances.csv"
        umap_file = "distance_data\\umap_distances.csv"
        pcatsne_file = "distance_data\\pcatsne_distances.csv"
        kmeans_centroid_file = "distance_data\\kmeans_centroid_distances.csv"

    def __init__(self):
        print("__init__ called")
        print("use_convnet:", self.use_convnet)
        print("id_key:", self.id_key)

    def getBatchTotal(self):
        print("use_convnet:", self.use_convnet)
        return self.batch_total
