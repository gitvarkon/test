import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import datasets

import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#%matplotlib inline

import seaborn as sns


from matplotlib.patches import Ellipse

def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)

## helper function to compute log(hypervolume)
def compute_log_hv(X):
    """
    Return log hypervolume defined as square-root of determinant of covariance matrix of d-dimensional sample.
    Input X covariance of sample
    """
    
    import numpy as np
        
    # compute eig values and take logprod
    w, _ = np.linalg.eig(X)
    log_w = np.log(w)/2
    log_hv = np.sum(log_w)
    
    return log_hv

## helper function to compute log(hypervol)
def compute_log_sum_hv(X):
    '''
    Given array of X = [log x1, log x2, ... , log xn]
    compute log(x1 + ... + xn)
    This problem arises in high dimensions (~ 100) when one needs to compute sum of hypervolume of clusters
    log was taken to prevent premature divergence to infinity or convergence to zero. 
    
    Starting with array sorted such that log x1 > log x2 > .... > log xn,
    we have
    
    log(x1 + ... + xn) 
    
    = log[x1 (x1 + x2 + ... + xn)/x1] 
    
    = log x1 + log[1 + x2/x1 + x3/x1 + ... + xn/x1]
    
    The 2nd term is extremely unlikely to diverge unless n is of order 10^10! 
    Do you have 10^10 clusters in your data? :)
    
    ~ log x1 + (x2/x1) + .... + (xn/x1) [taylor expansion to first order]
    
    = log x1 + log(1 + y)
    
    where y = exp(log x2 - log x1) + ... + exp (log xn - log x1)
    '''
    
    import numpy as np
    
    if len(X) == 1:
        log_sum_X = X
    else:
        X_sorted = np.sort(X)[::-1]
    
        max_X = X_sorted[0]
    
        y = [(X_sorted[i] - max_X) for i in range(len(X)) ]
        y = np.sum(np.exp(y))
    
        log_sum_X = max_X + np.log(y)
    #    log_sum_X = max_X + y
    
    return log_sum_X
    
'''
# visualization
def draw_digit(data, row, col, n):
    size = 28
    plt.subplot(row, col, n)
    plt.imshow(data)
    plt.gray()

## Example where $k$-means underfits data 
**$k$-means algorithm implicitly assumes**

[1] **circular** clusters 

    i.e. uncorrelated and equal variances across features

[2] **equal** cluster sizes

    i.e. risk of breakdown for data composed of 1000 samples from cluster 1, and 100 sample from cluster 2
    
Data consisting of clusters with variable sizes and shapes (especially with highly correlated, i.e. diagonally elongated clusters) gives rise to **model underfitting** with $k$-means! 

Example below: (4 clusters of variable sizes and shapes in 2-dimensional space, with both dimensions equally scaled) 
'''
# 1st cluster: mean, cov, number of samples
m1 = [-1,1]
cov1 = [[10,9.9],[9.9,10]]
n1 = 1000

# 2nd cluster: mean, cov, number of samples
m2 = [1,-1]
cov2 = [[10,9.9],[9.9,10]]
n2 = 1000

# 3nd cluster: mean, cov, number of samples
m3 = [5,-5]
cov3 = [[2,0],[0,2]]
n3 = 500

# 4th cluster: mean, cov, number of samples
m4 = [-5,5]
cov4 = [[2,0],[0,2]]
n4 = 500

# generate random samples
X1 = np.random.multivariate_normal(m1, cov1, n1)
X2 = np.random.multivariate_normal(m2, cov2, n2)
X3 = np.random.multivariate_normal(m3, cov3, n3)
X4 = np.random.multivariate_normal(m4, cov4, n4)

# concatenate all clusters
X = np.vstack((X1, X2, X3, X4))

# scatter plot
plt.plot(X[:,0], X[:,1], 'k.', ms=2)  # plotting black dots (k.) with marker size = 2 (smallest being 1)
plt.title("Fig 1:  Original data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
'''
- Generated data in 2-dimensional (feature) space to facilitate visualization, but similar argument extends to higher dimension.

- 4 clusters altogether, with 2 clusters highly correlated in features, and different sample sizes

- both axes with same ratio

### [1] $k$-means assuming number of clusters known (=4)
'''
# model selection: k-means
km = KMeans(n_clusters=4)
km.fit(X)
y_km = km.predict(X)

# plot results
plt.scatter(X[:, 0], X[:, 1], c=y_km, s=2, cmap="jet", edgecolors="none")  # colour scheme according y_km or predicted k-means partition, size=2, colourmap arbitrarily set to "jet"
plt.title("Fig 2:  Analysis of original data using $k$-means")
plt.xlabel("Elephant 1")
plt.ylabel("Feature 2")
plt.show()