'''
Created on Apr 1, 2016

@author: kartik
'''

import numpy as np
import urllib
from SVM.SVM import SVM
from Decision_Tree.Decision_Tree import Decision_Tree
import time 
def main():

    # url with dataset
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    # download the file
    raw_data = urllib.urlopen(url)
    # load the CSV file as a numpy matrix
    dataset = np.genfromtxt(raw_data,delimiter=",",dtype="i10,i2,i2,i2,i2,i2,i2,i2,i2,i2,i2",names=('id',
                                                                     'clump thickness',
                                                                     'uniformity of cell size',
                                                                     'uniformity of cell shape',
                                                                     'Marginal Adhesion',
                                                                     'Single Epithelial Cell Size',
                                                                     'Bare Nuclei',
                                                                     'Bland Chromatin',
                                                                     'Normal Nucleoli',
                                                                     'Mitoses',
                                                                     'Output'))
    # separate the data from the target attributes
    #print(dataset)
    data=dataset.tolist()
    output_list=[]
    alpha_list=[]
    for set in data:
        if set[10]==2:
            output_list.append(1)
        elif set[10]==4:
            output_list.append(-1)
        alpha_list.append(0)
    #splitting data into test and train data
    #initially splitting it into half
    training_index=len(data)/2
    regularization_parameter=7
    max_tolerance=pow(10,-5)
    max_passes=10
    
    print("\tRunning the SVM classifier:")
    print("\tLength of training data :"+str(training_index))
    print("\tParameters we are using for the SVM machine:")
    print("\tRegularization parameter: 7")
    print("\tMaximum tolerance:"+str(max_tolerance))
    print("\tMaximum passes: "+str(max_passes))
    
    svm =SVM(regularization_parameter,max_tolerance,max_passes,alpha_list,output_list, training_index, data)
    start=time.time()
    svm.Lagrange_multiplier()
    print("Threshold value(b) is "+ str(svm.b))
    print("After the training the alpha vector is :"+str(svm.alpha_list))
    print("Calculating the model for the test data:\n")
    print("Length of test data :"+str(len(data)-training_index))
    svm.predict_test_data()
    end=float(time.time())
    print ("Time taken by SVM classifier is:")
    print(str(end-start)+"s")
    #formatting the data
    data1=[list(row) for row in data]
    for row in data1 :
        if row[10]==2:
            row[10]=1
        else:
            row[10]=-1
    #data2=[[1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1, 1], [1002945, 5, 4, 4, 5, 7, 10, 3, 2, 1, 1], [1015425, 3, 1, 1, 1, 2, 2, 3, 1, 1, -1]]
    start=time.time()
    print("\tStarting the decision tree classifier")
    print("\tLength of training data:"+str(training_index))
    deci=Decision_Tree(data1,training_index)
    root_node=deci.buildtree([row for index,row in enumerate(data1) if index< training_index ])
    print("\tDisplaying the tree in In-order form")
    deci.display_tree(root_node)
    print("Calculating the model for the test data:\n")
    print("Length of test data :"+str(len(data)-training_index))
    deci.measure_accuracy(root_node)
    end=float(time.time())
    print ("Time taken by decision tree classifier is:")
    print(str(end-start)+"s")
    
    
main()
