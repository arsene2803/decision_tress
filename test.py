'''
Created on Apr 5, 2016

@author: kartik
'''
from SVM.SVM import SVM
xi=(5, 1, 1, 1, 2, 1, 3, 1, 1, 2)
xj=(5, 4, 4, 5, 7, 10, 3, 2, 1, 2)
svm=SVM(-7,8,9,[1,2,3],[-3,4,5],-7,[1,2,3])
sum=svm.kernal_function(xi,xj)
print(sum)


if __name__ == '__main__':
    pass