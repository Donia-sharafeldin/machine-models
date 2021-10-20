import numpy

import matplotlib.pyplot as plt
import numpy as np



class perceptron:
    def __init__(self,learning_rate = 0.01, number_of_iter = 1000):
        self.learning_rate = learning_rate
        self.number_of_iter = number_of_iter

    def fit(self,x,y):
            
            self.m , self.n = x.shape

            self.w = np.zeros(self.n)
            self.b =  0

            ypredicted = np.array([1 if i > 0 else 0 for i in y])
            for _ in range(self.number_of_iter):

                for i, val in  enumerate(x):
                    line = np.dot(val,self.w)+self.b
                    ypredicted = self.activation(line)
                    difference =self.learning_rate*(y[i]-ypredicted)
                    
                    self.w += difference * val
                    self.b += difference
         

    def activation(self,output):
        return np.where(output >= 0 ,1 ,0)
              
    def predict(self, x):
        return self.activation(np.dot(x,self.w)+ self.b)

 