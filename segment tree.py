# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:50:45 2019

@author: YuJeong
"""

import csv
from anytree import Node, RenderTree, findall, util
import string
import time
import numpy as np
import sys

#root == 1

######################### Taxonomy 생성  ###################################

def ReadCSV(filename):
    ff = open(filename, 'r', encoding = 'utf-8')
    reader = csv.reader(ff)
    headers = next(reader, None)
    data = {}

    for hh in headers:
        data[hh] = []
        
    for row in reader: 
        for hh, vv in zip(headers, row):
                data[hh].append(vv)
    return data

def GenerateItemHierarchyTree(treeItem):
    for i in range(len(treeItem['Name'])):
        globals()[treeItem['Name'][i]] =  Node(treeItem['Name'][i], parent = globals()[treeItem['Parent'][i]], data = treeItem['Data'][i])
        item_hierarchy_tree.append(globals()[treeItem['Name'][i]])
    return root

def PrintItemHierarchyTree(root):
    print("=="*30)
    for row in RenderTree(root):
        pre, fill, node = row
        print(f"{pre}{node.name}, data: {node.data}")
    print("=="*30)


############################# 오일러 투어 생성  ############################

#def EulerTour(treeIndex):

global no2serial
global serial2no
global locInTrip
global depth
global n
n = 13
no2serial = [-1 for i in range(n)]
serial2no = [-1 for i in range(n)]

locInTrip = [-1 for i in range(n)]
depth = [-1 for i in range(n)]

nextSerial = 0
def traverse(here, d, trip):
    global nextSerial

    print("nextSerial: ", nextSerial)
    print("here: " , here )
    print("d: ", d)
    print("trip size: ", len(trip) )

    for i in range(0, len(trip)):
        print(trip[i], end=' ')
    
    print("")
    print("*"*40)
    
   # global nextSerial
    no2serial[here] = nextSerial
    serial2no[nextSerial] = here
    nextSerial = nextSerial + 1
    
    depth[here] = d
    
    locInTrip[here] = len(trip)
    trip.append(no2serial[here])
    
    for i in range(0, len(child[here])):
      # nextSerial = nextSerial + 1
       # print("i: " , i)
       # print(" len(child[here]:",  len(child[here]))
        traverse(child[here][i], d + 1, trip )
        
        trip.append(no2serial[here])
    
        

def prepareRMQ():
  #  nextSerial = 0
    trip = []
    traverse(0, 0 , trip)


     
if __name__ == '__main__':
    
    treeItem = ReadCSV('C:/Users/YuJeong/Documents/Sequence-Similarity/eulerData.csv')
    item_hierarchy_tree = [] 

    root = Node("0", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)
    
  
    #print("후손 수:: " , len(root.descendants))
    queries = 3 #이 트리로 몇번 계산할지

    n = len(root.descendants)+1 #부모 노드 수
    parent = [0, 1, 1, 3, 3, 0, 6, 0, 8, 9, 9, 8]
    child = []
    for i in range(1, n+1):
        parentTemp = []
        for j in range(12):
            if i-1 == parent[j]:
                parentTemp.append(j+1)
        child.append(parentTemp)
    
    prepareRMQ()
    

    
    
    
    
    
  #  eulerTour = EulerTour(0, fc, ind)