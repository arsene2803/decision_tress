'''
Created on Apr 2, 2016

@author: kartik
'''
import math
import random


class SVM(object):
    '''
    Implementing the algorithm here
    '''


    def __init__(self,regularization_parameter,numerical_tolerance,max_passes,alpha_list,output_list,training_index,data):
        '''
        Constructor
        '''
        self.C=regularization_parameter
        self.tol=numerical_tolerance
        self.max_passes=max_passes
        self.training_index=training_index
        self.output_list=output_list
        self.alpha_list=alpha_list
        self.data=data
        self.weight_vector=[]
        self.b=0
        
    def kernal_function(self,xi,xj):
        sigma=8
        sum=0
        for i in range(1,len(xi)-1):
            sum+=math.exp(-(math.pow(xi[i]-xj[i], 2))/(2*math.pow(sigma,2)))
        return sum   
            
            
                
        
    def feature_function(self,index,b):#f(x)
        
        value=0
        for i in range(self.training_index):
            value=self.alpha_list[i]*self.output_list[i]*self.kernal_function(self.data[i],self.data[index])+b
        
        return value
        
        
        
        
        
    def error_function(self,index,b):
        value=self.feature_function(index, b)-self.output_list[index]
        return value        
    
    
    def Lagrange_multiplier(self):
        passes=0
        b=0 #threshold value
        while(passes<self.max_passes):
            number_changed_alphas=0
            for i in range(self.training_index):
                ai=self.alpha_list[i]#alpha i
                yi=self.output_list[i]
                ei=self.error_function(i, b)#Ex(i)
                xi=self.data[i]
                if(yi*ei<(-self.tol)and ai< self.C) or (yi*ei>self.tol and ai> 0):
                    
                    j=random.randint(0,self.training_index-1)
                    while(j==i):
                        j=random.randint(0,self.training_index-1)
                    aj=self.alpha_list[j]#alpha j
                    xj=self.data[j]
                    yj=self.output_list[j]
                    ej=self.error_function(j, b)
                    temp_alpha_i=ai
                    temp_alpha_j=aj
                    if(yi!=yj):
                        L=max(0,aj-ai)
                        H=min(self.C,self.C+aj-ai)
                    if(yi==yj):
                        L=max(0,aj+ai-self.C)
                        H=min(self.C,aj+ai)
                    if(L==H):
                        continue
                    eita=2*self.kernal_function(xi,xj)-self.kernal_function(xi,xi)-self.kernal_function(xj,xj)
                    if(eita>=0):
                        continue
                    aj=aj-yj*(ei-ej)/eita
                    if aj>H:
                        aj=H
                    elif aj<L:
                        aj=L
                    if abs(aj-temp_alpha_j)< pow(10,-5):
                        continue
                    
                    ai=ai+yi*yj*(temp_alpha_j-aj)
                    
                    self.alpha_list[i]=ai
                    
                    b1=b-ei-yi*(ai-temp_alpha_i)*self.kernal_function(xi,xi)-yj*(aj-temp_alpha_j)*self.kernal_function(xi,xj)
                    
                    b2=b-ej-yi*(ai-temp_alpha_i)*self.kernal_function(xi,xj)-yj*(aj-temp_alpha_j)*self.kernal_function(xj,xj)
                    
                    if ai >0 and ai < self.C:
                        b=b1
                    elif aj>0 and aj<self.C:
                        b=b2
                    else:
                        b=(b1+b2)/2
                    number_changed_alphas+=1
            if number_changed_alphas==0:
                passes+=1
        self.b=b
                 
    
       
    def predict_test_data(self):
        output_test=[]
        inaccurate_count=0
        
        for i in range(self.training_index,len(self.data)):
            f_x=0
            for j in range(self.training_index):
                f_x+=self.alpha_list[j] * self.output_list[j] *self.kernal_function(self.data[j],self.data[i])
            if f_x>=0:
                output_test.append(1)
            else:
                output_test.append(-1)
        for i in range(self.training_index,len(self.data)):
            if self.output_list[i] != output_test[i-self.training_index]:
                inaccurate_count+=1                        
        print inaccurate_count
        print len(self.data)-self.training_index
        accuracy=(1-float(inaccurate_count)/(len(self.data)-self.training_index))*100
        print(accuracy)
        print("Accuracy of the SVM on test data:"+str(accuracy))          
            
                     
                           
                    
        
                    
                    
                        
                
                
                
            
           
           