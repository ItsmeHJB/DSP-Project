import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow import keras

# from keras.datasets import cifar10
# from keras.datasets import mnist
# from keras.datasets import fashion_mnist

from sys import platform


def get_label_array(argument):
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

    batch_size = 0
    batch_total = 0
    test_size = 0
    total_test_pool = 10000
    total_train_pool = 55000

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
    x_train = np.zeros([0, 784])
    y_train = np.zeros([0, 10])
    x_test = []  # np.zeros([1,784])
    y_test = []  # np.zeros([1,10])

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

    #### Can we get it working so that we cn drop in new datasets easily?
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Load mnist data set
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape
    x_train = np.delete(x_train, np.s_[55000:], axis=0)
    x_train = x_train.reshape(55000, 784)
    x_test = x_test.reshape(10000, 784)

    # Convert new style labels back to binary array of [0,9]
    temp = []
    y_train = np.delete(y_train, np.s_[55000:], axis=0)
    for value in y_train:
        temp.append(get_label_array(value))

    y_train = np.array(temp)

    del temp[:]
    for value in y_test:
        temp.append(get_label_array(value))

    y_test = np.array(temp)

    if running_on_osx:
        directory_for_test_images = "static/images/mnist_noclass/test"
        directory_for_train_images = "static/images/mnist_noclass/train"
        directory_for_trained_model = "logs/trained_logreg_model.ckpt"

        tsne_file = "./distance_data/tsne_distances.csv"
        pca_file = "./distance_data/pca_distances.csv"
        umap_file = "./distance_data/umap_distances.csv"
        pcatsne_file = "./distance_data/pcatsne_distances.csv"
        kmeans_centroid_file = "./distance_data/kmeans_centroid_distances.csv"
    else:
        directory_for_test_images = "static\\images\\mnist\\test"
        directory_for_train_images = "static\\images\\mnist\\train"
        directory_for_test_images = "static\\images\\mnist_noclass\\test"
        directory_for_train_images = "static\\images\\mnist_noclass\\train"
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
