import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Linear_Regression:

    def __init__(self, lrate = 0.001, iterations= 100):
        self.lrate = lrate
        self.iterations = iterations
        self.weights = None
        self.bias = None



    def fit(self, X, y):

        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features) # initializing the weights =0 and bias
        self.bias = 0.01

        #implementing gradient descent

        for i in range(self.iterations):

            y_predicted = np.dot(X, self.weights) + self.bias

            dw =(2/num_samples)*np.dot(X.T, (y_predicted-y)) # calculating the gradient
            db =(2/num_samples)*np.sum(y_predicted-y)

            self.weights = self.weights -self.lrate*dw # updating the weights and bias
            self.bias = self.bias -self.lrate*db




    def predict(self,X):
        y_predicted = np.dot(X, self.weights) + self.bias

        return y_predicted

#creating sample data
X,y = datasets.make_regression(n_samples = 200, n_features =1, random_state= 3)
X_train,X_test,y_train,y_test =train_test_split(X,y, test_size =0.2,random_state =15)

#
lr = Linear_Regression(lrate = 0.002)
lr.fit(X_train,y_train)

predicted_op = lr.predict(X_test)

# Performance meausing MSE
mse = np.mean((y_test -predicted_op)**2)
print("The Mean Squared Error is : ", mse)

