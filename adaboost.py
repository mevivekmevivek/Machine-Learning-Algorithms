
#----------------------------------------------------------------------------------------------------------------
# In[1]:


# importing the required libraries
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#----------------------------------------------------------------------------------------------------------------
# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm

#m1 = LogisticRegression()

# we will define the adaboost class
class Ada_Boosting:
    
    
    #def __init__(self,number_of_classfiers=50, base_estimator=None):
        
     #   self.number_of_classfiers = number_of_classfiers
        
      #  if base_estimator == None:
       #     base_estimator = DecisionTreeClassifier(max_depth=1)
        
        #self.base_estimator = base_estimator
        
    
    # constructor of the adaboosting class with just one argument (number of classifiesr)
    def __init__(self, number_of_classfiers):
        
        self.number_of_classfiers = number_of_classfiers
        
    # defining the fit method- (the method takes in train data and the target variable as arguments)
    def fit(self, traindata, targetvariable):
        
        
        numberOfSamples =  traindata.shape[0]
        numberOffeatures = traindata.shape[1]
        
        self.classifiers = []  # empty lists for saving the classifier objects and alpha values
        self.alphas = []
        
        w_weights = np.ones(numberOfSamples) / numberOfSamples # initializing the weights to 1/numberOfSamples
        
        #looping through the classifiers
        
        i =0
        while i <= self.number_of_classfiers:
            
            
            #Gm = clone(self.base_estimator).\
                            #fit(traindata,targetvariable,sample_weight=w_weights).predict
        
            #predictions = Gm(traindata)
            
            #error = w_weights.dot( predictions != targetvariable )
            
           # tree = LogisticRegression()
            
            #tree = svm.SVC(kernel ='linear', gamma = 'auto', C =  3)
            
            tree = DecisionTreeClassifier(max_depth =1)
            tree.fit(traindata,targetvariable,sample_weight = w_weights)
            predictions = tree.predict(traindata)
            
            # calculating the error and alpha values
            
            error = w_weights.dot( predictions != targetvariable )
            alpha = 0.5*(np.log(((1-error)+ (1e-10))/(error + (1e-10)))) 
                
            
            #updating the weights and normalizing
            w_weights = w_weights * np.exp(-1*alpha*targetvariable*predictions)
            
            w_weights = w_weights/w_weights.sum()
            
            # saving the classifier and alpha 
            
            self.classifiers.append(tree)
            self.alphas.append(alpha)
            
            i = i+1
                                 
            
# defining the predict method of the adaboost class

    def predict(self,traindata):
        
        numberOfSamples =  traindata.shape[0]
        numberOffeatures = traindata.shape[1]
        
        HX = np.zeros(numberOfSamples)
        
        for alpha, tree in zip(self.alphas, self.classifiers):
            HX+= alpha*tree.predict(traindata)
        return np.sign(HX)
    
    def loss(self,traindata, y):
        
        numberOfSamples =  traindata.shape[0]
        numberOffeatures = traindata.shape[1]
        
        HX = np.zeros(numberOfSamples)
        
        for alpha, tree in zip(self.alphas, self.classifiers):
            HX+= alpha*tree.predict(traindata)
        
        L = np.exp(-y*HX).mean()
        
        return L 

#----------------------------------------------------------------------------------------------------------------
# In[3]:


if __name__ == '__main__':  
    
   # importing the data and getting it ready
    train = pd.read_csv("C://Users//vivek//Desktop//bigdata//wdbc_data.csv", header = None)
    y_tr = train[1]
    x_tr = train.drop([0,1], axis =1)
    x_tr.shape 
    x = np.array(x_tr.iloc[:,:])
    y = np.array(y_tr)
    y[y == 'B'] = 1 
    y[y == 'M'] = -1 
    y=y.astype('int')
    
    
    from sklearn.model_selection import train_test_split

    X_train, X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.472759227 )
    
    
    #tree = LogisticRegression()
    
    #tree = svm.SVC(kernel ='linear', gamma = 'auto', C =  3)    

    tree = DecisionTreeClassifier(max_depth =1)

    numberofmodels  = 200
    
    trainingerrors = np.empty(numberofmodels)
    
    testingerrors = np.empty(numberofmodels)
    
    testlosses = np.empty(numberofmodels)
    
        
    qq = 0
        
    for model in range(numberofmodels):
        
        if model == 0:
            
            trainingerrors[model] = None
            testingerrors[model] = None
            
            continue

        #if model %50 ==0:
         #   print(model)
            
        basemodel = Ada_Boosting(model )
        basemodel.fit(X_train, Y_train)
        
        Predictions= basemodel.predict(X_test)        
        accuracy = np.mean( Predictions ==Y_test)
        
        trainingPredictions =  basemodel.predict(X_train)
        trainingaccuracy = np.mean( trainingPredictions ==Y_train)
        
        trainingerrors[model] = 1 - trainingaccuracy
        testingerrors[model] = 1 - accuracy
        testlosses[model] = basemodel.loss(X_test, Y_test )
        
                
        if model  == 1:
            
            print("Intial train error:", 1 - trainingaccuracy)
            print("Intial test error:", 1 - accuracy)
            print()
            
            
        if (trainingaccuracy ==1) and (qq ==0):    
                
            print("The train error becomes Zero at :", model)
            print("At the point when train error is Zero, the test error:", 1 - accuracy)
            print()                
            qq = 1 
            
        if model == numberofmodels - 1:
            
            print("Final train error:", 1 - trainingaccuracy)
            print("Final test error:", 1 - accuracy, "Test accuracy :", accuracy*100 )
                
    
    plt.plot(trainingerrors, label='train errors')
    plt.plot(testingerrors, label='test errors')
    plt.legend()
    plt.show()
    
    plt.plot(basemodel.alphas, label='Alpha')
    plt.legend()
    plt.show()
    
    #plt.plot(testingerrors, label='test errors')
    #plt.plot(testlosses, label='test losses')
    #plt.legend()
    #plt.show()
#----------------------------------------------------------------------------------------------------------------

# In[4]:


# comparision with SVM

from sklearn import svm
svmmodel = svm.SVC(kernel ='linear', gamma = 'auto', C =  1)
svmmodel.fit(X_train, Y_train)


y_predicticted = svmmodel.predict(X_test)
svmaccuracy = np.mean( y_predicticted ==Y_test)
print("SVM accuracy : ", svmaccuracy*100,"Adaboost accuracy : ",accuracy*100)

#----------------------------------------------------------------------------------------------------------------
# In[7]:


#comparision with AdaBoost Inbuilt

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

l1 = []

num_estimators = np.array([100, 200, 300, 400, 500, 600, 700,800,900,1000, 1100, 1200,1300,1400,1500])

for num_estimator in num_estimators:
    
    Adamodel = AdaBoostClassifier(n_estimators=num_estimator, learning_rate=1)

    estimator = Adamodel.fit(X_train, Y_train)


    Y_pred= estimator.predict(X_test)
    
    print("Number of estimators: ",num_estimator, "Accuracy: " , metrics.accuracy_score(Y_pred, Y_test))
    l1.append(metrics.accuracy_score(Y_pred, Y_test))
   
#estimator.predict(X_test) 

plt.plot(estimator.estimator_errors_ , label='Error')
plt.legend()
plt.show()
    
plt.plot(l1 , label='Accuracy')
plt.legend()
plt.show()

#----------------------------------------------------------------------------------------------------------------





