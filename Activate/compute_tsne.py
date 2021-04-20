import numpy as np
import os

from activate_config import Options as oo
from PIL import Image


X = np.array(oo.train_images)
Y = np.array(oo.train_labels)
actual_labels = Y.argmax(axis=1).reshape([X.shape[0], 1])
user_labels = np.ones([actual_labels.shape[0], 1]) * -1
user_confidences = np.zeros([actual_labels.shape[0], 1])
loaded = False

metric = 'pca_tsne'

if metric == 'pca':
    if os.path.exists(oo.pca_file):
        print ("Loading PCA from file...")
        oo.pca_distances = np.loadtxt(oo.pca_file, delimiter=',')
        loaded = True
    else:
        print ("Computing PCA... this can take some time...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(X)
        oo.pca_distances = pca.transform(X)
        this_filename = oo.pca_file
elif metric == 'umap':
    if os.path.exists(oo.umap_file):
        print ("Loading UMAP from file...")
        oo.pca_distances = np.loadtxt(oo.umap_file, delimiter=',')
        loaded = True
    else:
        print ("Computing UMAP... this can take some time...")
        import umap
        oo.pca_distances = umap.UMAP(n_neighbors=10, min_dist=0.001, metric='correlation').fit_transform(X)
        this_filename = oo.umap_file

elif metric == 'tsne':
    if os.path.exists(oo.tsne_file):
        print ("Loading TSNE from file...")
        oo.pca_distances = np.loadtxt(oo.tsne_file, delimiter=',')
        loaded = True
    else:
        print ("Computing TSNE... this can take some time...")
        from sklearn.manifold import TSNE
        oo.pca_distances = TSNE(n_components=2, verbose=1).fit_transform(X)
        this_filename = oo.tsne_file
        print ("this_filename:", this_filename)

elif metric == 'pca_tsne':
    if os.path.exists(oo.pcatsne_file):
        print ("Loading PCA-TSNE from file...")
        oo.pca_distances = np.loadtxt(oo.pcatsne_file, delimiter=',')
        loaded = True
    else:
        print ("Computing PCA-TSNE... this can take some time...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        pca.fit(X)
        oo.pca_distances = pca.transform(X)
        from sklearn.manifold import TSNE
        oo.pca_distances = TSNE(n_components=2, verbose=1).fit_transform(oo.pca_distances)
        oo.pca_distances = np.hstack([oo.pca_distances, actual_labels])
        this_filename = oo.pcatsne_file
        print ("this_filename:", this_filename)

elif metric == 'kmeans_centroids':
    if os.path.exists(oo.kmeans_centroid_file):
        print ("Loading kMeans Centroid from file...")
        oo.pca_distances = np.loadtxt(oo.kmeans_centroid_file, delimiter=',')
        loaded = True
    else:
        print ("Computing kMeans Centroid... this can take some time...")
        from sklearn.cluster import KMeans
        oo.pca_distances = KMeans(n_clusters=10, random_state=0).fit_transform(X)
        from sklearn.manifold import TSNE
        oo.pca_distances = TSNE(n_components=2, verbose=1).fit_transform(oo.pca_distances)
        oo.pca_distances = np.hstack([oo.pca_distances, actual_labels])
        this_filename = oo.kmeans_centroid_file
        print ("this_filename:", this_filename)

print ("loaded", loaded)
if not loaded:
    if oo.create_actual_label_column:
        oo.pca_distances = np.hstack([oo.pca_distances, actual_labels])

    if oo.create_predicted_label_column:
        from sklearn.cluster import KMeans
        clf = KMeans(n_clusters = oo.number_of_classes, random_state=42)
        clf.fit(X)
        labels = clf.labels_
        labels = labels.reshape([oo.pca_distances.shape[0], 1])
        oo.pca_distances = np.hstack([oo.pca_distances, labels])

    if oo.create_user_label_column:
        oo.pca_distances = np.hstack([oo.pca_distances, user_labels])

    if oo.create_user_confidence_column:
        oo.pca_distances = np.hstack([oo.pca_distances, user_confidences])

    oo.pca_distances[:,0] = ((oo.pca_distances[:,0] - np.min(oo.pca_distances[:,0])) / (np.max(oo.pca_distances[:,0]) - np.min(oo.pca_distances[:,0])) * 2) - 1
    oo.pca_distances[:,1] = ((oo.pca_distances[:,1] - np.min(oo.pca_distances[:,1])) / (np.max(oo.pca_distances[:,1]) - np.min(oo.pca_distances[:,1])) * 2) - 1
    print ("Writing to file:", this_filename)
    np.savetxt(this_filename, oo.pca_distances, delimiter=',')

index_count = np.array(range(oo.pca_distances.shape[0])).reshape(oo.pca_distances.shape[0],1)
oo.pca_distances = np.hstack([index_count,oo.pca_distances])
oo.pca_distances_working_copy = np.copy(oo.pca_distances)

oo.normal_distances = np.hstack([index_count,X])
oo.normal_distances_working_copy = np.copy(oo.normal_distances)

print ("ALL DONE! Metric:", metric)
