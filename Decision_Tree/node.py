'''
Created on Apr 6, 2016

@author: kartik
'''

class node(object):
    '''
    classdocs
    '''


    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col=col # column index of criteria being tested
        self.value=value # vlaue necessary to get a true result
        self.results=results # dict of results for a branch
        self.tb=tb # true decision nodes 
        self.fb=fb # false decision nodes
        
 
        
        
        
        