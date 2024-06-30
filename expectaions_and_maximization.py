import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
def draw_ellipse(position, covariance, ax):
    if covariance.shape == (2, 2):
        U, s, _ = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        width, height = 2 * np.sqrt(covariance)
        angle = 0
    ellipse = plt.Circle(position, width, alpha=0.2, color='red')
    ax.add_patch(ellipse)
ax = plt.gca()
for pos, covar in zip(gmm.means_, gmm.covariances_):
    draw_ellipse(pos, covar, ax)
plt.show()
