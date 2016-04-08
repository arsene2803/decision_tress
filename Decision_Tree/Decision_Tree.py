'''
Created on Apr 6, 2016

@author: kartik
'''
from node import node
from collections import defaultdict

class Decision_Tree(object):
    '''
    classdocs
    '''


    def __init__(self,data,training_index):
        '''
        Constructor
        '''
        self.data=data
        self.training_index=training_index

        

    def result_count(self,data):
        results = defaultdict(lambda: 0)
        for row in data:
            r = row[len(row)-1]
            results[r]+=1
        return dict(results)
    
    

    def divideset(self,data,column,value):
     
        split_function=None
        # for numerical values
        split_function=lambda row:row[column]>=value
       
        # Divide the data into two sets and return them
        set1=[row for row in data if split_function(row)] # if split_function(row) 
        set2=[row for row in data if not split_function(row)]
        return (set1,set2)
    
    def entropy(self,data):
        
        from math import log
        log2=lambda x:log(x)/log(2)
        result=self.result_count(data)
        value=0
        p=[]
        for key in result.keys():
            p.append(float(result[key])/len(data))
        for item in p:
            value+=-(item*log2(item))
        return value    
    
    def buildtree(self,data):
        if len(data) == 0: return node()
        
        current_entropy = self.entropy(data)
        best_gain = 0.0
        best_criteria = None
        best_sets = None
    
        
        for col in range(1,10):
            # find different values
            column_values = set([row[col] for row in data])
    
            # for each possible value, try to divide on that value
            for value in column_values:
                true_set, false_set = self.divideset(data,col,value)
                #===============================================================
                # print("true set:"+str(true_set))
                # print("false_set:"+str(false_set))
                # print ("column"+str(col))
                # print("value"+str(value))
                #===============================================================
                # Information gain
                p = float(len(true_set)) / len(data)
                gain = current_entropy - p*self.entropy(true_set) - (1-p)*self.entropy(false_set)
                
                if gain > best_gain and len(true_set) > 0 and len(false_set) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (true_set, false_set)
        print "best gain: "+str(best_gain)
        print "best criteria: "+str(best_criteria)
        print "best sets: "+str(best_sets)
        if best_gain > 0:
            trueBranch = self.buildtree(best_sets[0])
            falseBranch = self.buildtree(best_sets[1])
            return node(col=best_criteria[0], value=best_criteria[1],
                    tb=trueBranch, fb=falseBranch)
        else:
            return node(results=self.result_count(data))        
    
    def display_tree(self,root_node):
        
        if root_node==None:
            return
        self.display_tree(root_node.tb)
        print "column: "+str(root_node.col)+" Value:"+str(root_node.value)+" Probability:"+str(root_node.results)
        self.display_tree(root_node.fb)  
        
    def measure_accuracy(self,root_node):
        print("Calculating the accuracy of the decision tree for the test data:")
        inaccurate_count=0
        prev_node=None
        test_data=[row for index,row in enumerate(self.data) if index>= self.training_index ]
        for row in test_data:
            curr_node=root_node
            while curr_node is not None:
                prev_node=curr_node
                if row[curr_node.col]>=curr_node.value:
                    curr_node=curr_node.tb
                else:
                    curr_node=curr_node.fb
            max_key=max(prev_node.results, key=lambda i: prev_node.results[i])
            if max_key!=row[len(row)-1]:
                inaccurate_count+=1
        accuracy=(1-float(inaccurate_count)/(len(self.data)-self.training_index))*100
        print("Accuracy of the decision tree classifier on test data:"+str(accuracy))          
                
                
                
                
                
            
            
        
        
            