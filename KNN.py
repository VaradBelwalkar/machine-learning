import numpy as np
from collections import Counter
#Here we are just calculating the euclidean distance 
#Understand that it is different that normal distance calculation that uses two variables(x,y) to calculate
#Here we just calculate on the basis of values of independent variable which is nothing but the FEATURE
def euclidean_distance(x1,x2):
    #Here rather than traditional distance calculation like sqrt( (x1-x2)**2 + (y1-y2)**2 )
    # We just use sqrt( (x1-x2)**2 ) which is abvious as y is label here which we need to predict
    # So considering y is not important here
    #Here just calculating x1-x2 would do the job but, the template follows 
    #feature vector that may contain multiple features, so adding them over here is easy
    distance  = np.sqrt(np.sum((x1-x2)**2))
    return distance 

#Here we are considering a default value for k as 3
class KNN:
    def __init__(self,k=3):
        self.k = k
    
    #This fit function does nothing but just stores the individual features and labels associated with them
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
        #Here the np.argsort isn't actually sorting the distances list
        #What it is doing is it just considers the sorted list, which it does
        #internally, and stores the indices of those values in the actual List
        #Example --- >
        #suppose we have list [4,5,2,8,12,1]
        #When we pass this array to the np.argsort, it generates this result
        #
        #               [5,2,0,1,3,4] 
        #
        #This is because, sorted list is [1,2,4,5,8,12]
        #So, it returns the index of 1 in actual list i.e 5, then for 2, it returns,
        #for 4 it returns 0 and so on

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #Get the most common label so it will be considered as prediction for the particular value
        
        most_common = Counter(k_nearest_labels).most_common()
        return most_common
