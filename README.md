# Machine-Learning ALgorithms
Some of the commonly used machine learning algorithms are developed from the scratch using basic libraries


Algorithm : Adaboost

An algorithmic description of the AdaBoost method
Boosting- Boosting is a technique in which a certain weak learning algorithm is scaled up to make the 
training error zero. In general, boosting is not a single algorithm it is a family of algorithms.

H(x)- Is the strong learner algorithm built by boosting, h(x)- is the weak learning algorithm

H(x) = sigma(k=1 to T)(alpha(i)* h(x(i))
T- Defines the number of weak classifiers
The weak learner h (x) is implemented on the given training set S = (x1, y1),(x2, y2), . . . ,(xn, yn) where ∀i, yi 
∈ {−1, 1} and T such weak learning algorithms are put together to build the strong learner H(x).

 
l(H) is the loss function which is convex and differentiable.
Adaboost- Adaboost (Adaptive Boosting) is a very specific instance of boosting 
1) where the loss function is chosen as l (below equation (2))
2) optimal value of α is chosen by performing a line search.
 
 
 
 
 Pseudo Code :
Fit (train data, target variable): 
 Step1- Classifiers = [], Alphas = [] (initializing empty classifier and alpha lists).
 
 Step2- wi = Initializing the weights to 1/n
 
 Step3- For (all the number of classifiers):
 – Fit the week learner (h) for the train data (train data, target variable, sample weight)
 
 Step4- 
 Compute error = Sum (wi*I(yi!=h(xi) ) / Sum (wi)
 Compute α = 0.5*(log((1-error)/(error)))
 Update the weights wi = wi * exp(α*I(yi!=h(xi)
 
 Step5- Saving the classifier (h) and alphas

Predict:
 H(x) = sign ( Sum (α*h(x)) )
 
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Algorithm: PCA

1) An algorithmic description of PCA
Principal component analysis is primarily used to reduce the dimensions of the dataset if the number of 
variables in a dataset is huge. Generally PCA outputs principal components which are the linear 
combination of optimally-weighted observed variables. Principal components are eigenvectors and 
these eigenvectors satisfy the principle of east squares. Below are the steps that need to be followed 
for PCA implementation.


Covariance(Data):
Step1- form a zero matrix of the shape of data
Step2_ compute mean and norm
Step 3-update each element of the above formed matrix

PCA_ Train (Data, Reduced number of dimensions ):
Step1- Calculate the mean, covariance and C
Step2- calculate the eigen values and eigen vectors
Step3-sort the eigen values and select the top eigen values and corresponding vector

Step4_ compute the project matrix and return the mean, eigen vectors, Project matrix and eigen 
values. 
PCA_ Transform (Data, mean, eigen vectors ):
Step1- Calculate C
Step2- project the data using eigen vectors and C

------------------------------------------------------------------------------------------------------------------------------------------------------------------

Algorithm : Kmeans

An algorithmic description of KMEANS
Kmeans is an unsupervised learning technique, the basic idea begin the algorithm is to group the data 
into various clusters. Below are the steps of the Kmeans algorithm.

1. Select K centroids randomly (Centroids can also be selected with some conditions)
2. Compute the Euclidian distances of all the data points from these centroids. 
3. Assign the data points to their closest clusters based on the Euclidian distances calculated.
4. Compute the new centroids of the clusters simply by calculating the mean of each cluster.
5. Repeat the above steps until the centroids stop moving.

In the first step for assigning the clusters, we need to calculate the distances of each randomly selected 
centroid to all the data points in the dataset. While updating the centroids in the step 4, as we have 
already assigned the clusters, we calculate the distances of the centroid from all the data points in their 
respective clusters.

Major Limitations of Kmeans

1. It is not possible to know the number of clusters or K in all the cases.
2. Kmeans is very sensitive to outliers as the entire algorithm depends on the calculation of 
distances and mean.
3. Kmeans is hard clustering algorithm, that is, it does not gives the probabilities with which each 
point belong to a cluster. It rather just says if the point belongs to a cluster or not. Hard 
clustering will result in loss of informatio
