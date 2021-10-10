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
2) optimal value of α is chosen by performing a line search 
 
 
 
 
 Pseudo Code :
Fit (train data, target variable): 
 Step1- Classifiers = [], Alphas = [] (initializing empty classifier and alpha lists)
5
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
 
 
Working- Adaboost classification algorithm works as the following
 Firstly, a random subset of the training data is selected by the Adaboost algorithm.
 The Adaboost algorithm is trained iteratively by selecting the train dataset based on the accurate 
predictions from the previous iteration. 
 All the misclassified points are assigned with higher weights such that these wrongly classified points 
are given higher priority in the next iteration.
 Also, based on the accuracy of the classifier, weights are assigned even to the classifier in each 
iteration. Higher the accuracy of a classifier, greater will be the weight assigned.
 The above steps will continue till the entire training data is fitted and the training error is zero.
 Classification is done by performing a vote across all the built learning algorithms.
