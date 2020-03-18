import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#LOAD AND PREPROCESS DATA

digits = load_digits()
data = scale(digits.data)#data is a npArray
#scale function to scale our data down.
#We want to convert the large values that are contained as 
#features into a range between -1 and 1 to simplify calculations 
# and make training easier and more accurate.

y = digits.target#y is a npArray
k = len(np.unique(y)) #k --> number of different "groups"

#n rows(one row includes the data of one digit) /m colums(one digit exists of 64 columns)
samples, features = data.shape 


#show digits 
for i in range(3):
    plt.matshow(digits.images[i])
    plt.show()

#shows model benchmarks
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (name, estimator.inertia_,
            metrics.homogeneity_score(y, estimator.labels_),
            metrics.completeness_score(y, estimator.labels_),
            metrics.v_measure_score(y, estimator.labels_),
            metrics.adjusted_rand_score(y, estimator.labels_),
            metrics.adjusted_mutual_info_score(y,  estimator.labels_),
            metrics.silhouette_score(data, estimator.labels_,metric='euclidean')))

#n_init:Number of time the k-means algorithm will be run with different centroid seeds.
model = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(model, "randomModel", data)
model = KMeans(n_clusters=k, init="k-means++", n_init=10)
bench_k_means(model, "k-means++Model", data)


#PLOT DATA AND PREDICITON

reduced_data = PCA(n_components=2).fit_transform(data)
#project the data into two dimensions
#PAC:
    #Linear dimensionality reduction using Singular Value Decomposition 
    #of the data to project it to a lower dimensional space.

fig, (ax1, ax2) = plt.subplots(1, 2,num='Plot',tight_layout=True)

#Plot data
ax1.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
ax1.set_title("Original data")
#Plot predicition /no idea how this works(for sklearn)
kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
kmeans.fit(reduced_data)
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax2.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower')
ax2.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
ax2.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
ax2.set_title('K-means clustering on the digits dataset')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_xticks(())
ax2.set_yticks(())

plt.show()