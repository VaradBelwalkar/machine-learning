import numpy as np
from collections import Counter
#Here we are just calculating the euclidean distance 
#Understand that it is different that normal distance calculation that uses two variables(x,y) to calculate
#Here we just calculate on the basis of values of independent variable which is nothing but the FEATURE
def euclidean_distance(x1,x2):
    #Here rather than traditional distance calculation like sqrt( (x1-x2)**2 + (y1-y2)**2 )
    # We just use sqrt( (x1-x2)**2 ) which is abvious as y is label here which we need to predict
    # So considering y is not important here
    distance  = np.sqrt(np.sum((x1-x2)**2))
    return distance 

#Here we are considering a default value for k as 3
class KNN:
    def __init__(self,k=3):
        self.k = k
    
    #This fit function does nothing but just stores the feature vector and labels
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y
    
    
    #Actual work and computation happens in the predict function
    
    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        #here We computer the distance for each value provided
        distances = [euclidean_distance(x,x_train) for x_train in self.x_train]

        #Get the closest K 
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(most_common = k_nearest_labels).most_common()
        return most_common
