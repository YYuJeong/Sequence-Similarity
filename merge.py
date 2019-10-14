# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:40:15 2019

@author: YuJeong
"""

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

def rootToZero(treeItem):
    parent = []
    for i in range(len(treeItem['Parent'])):
        if treeItem['Parent'][i] == 'root':
            parent.append(0)
        else:
            parent.append(int(treeItem['Parent'][i]))
    return parent
############################# 오일러 투어 생성  ############################


global no2serial
global serial2no
global locInTrip
global depth
global k
global rangeMin
k =  len(root.descendants)+1
no2serial = [-1 for i in range(k)]
serial2no = [-1 for i in range(k)]

locInTrip = [-1 for i in range(k)]
depth = [-1 for i in range(k)]

nextSerial = 0
def traverse(here, d, trip):
    global nextSerial
    global no2serial
    global serial2no
    global locInTrip
    global depth

    no2serial[here] = nextSerial
    serial2no[nextSerial] = here
    nextSerial = nextSerial + 1
    
    depth[here] = d
    
    locInTrip[here] = len(trip)
    trip.append(no2serial[here])
    
    for i in range(0, len(child[here])):
        traverse(child[here][i], d + 1, trip )
        
        trip.append(no2serial[here])

global count   
count = 0
def RMQ(array):
    global rangeMin
    global n
    n = len(array)
    rangeMin = []
    rangeMin = [0 for i in range(n*4)]   
    init(array, 1, 0 , n-1)

def init(array, node, left, right):
    if left == right: 
        rangeMin[node] = array[left]
        return rangeMin[node]
    mid = int((left + right)/2)
    rangeMin[node] = min(init(array, node * 2 + 1, mid + 1, right), init(array, node * 2, left, mid))
    return rangeMin[node]



def prepareRMQ():
    trip = []
    traverse(0, 0 , trip)
    return RMQ(trip)


def distance(u, v):
    lu = locInTrip[u]
    lv = locInTrip[v]
    if(lu > lv):
        lu, lv = lv, lu
    lca = serial2no[query1(lu, lv)]
    return depth[u]+depth[v]-2*depth[lca]


def query1(left, right):
    global n
    return query(left, right , 1, 0 , n-1)

def query(left, right, node, nodeLeft, nodeRight):
    global count
    count = count + 1
    if (right < nodeLeft) or (nodeRight < left):
        return sys.maxsize
    if (left <= nodeLeft) and (nodeRight <= right):
        return rangeMin[node]
    mid = int((nodeLeft + nodeRight)/2)
    return min(query(left, right, node * 2 + 1, (mid+1), nodeRight), query(left, right, node * 2, nodeLeft, mid))
    
  
     
if __name__ == '__main__':
    
    treeItem = ReadCSV('C:/Users/YuJeong/Documents/Sequence-Similarity/eulerData.csv')
    item_hierarchy_tree = [] 

    root = Node("0", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)
    
  
    print("후손 수:: " , len(root.descendants))
    queries = 3 #이 트리로 몇번 계산할지
    
    n = len(root.descendants)+1 #부모 노드 수
    parent = rootToZero(treeItem)
#    n = 13
    #parent = [ 0, 1, 1, 0 ,4]
    child = []
    for i in range(1, n+1):
        parentTemp = []
        for j in range(len(parent)):
            if i-1 == parent[j]:
                parentTemp.append(j+1)
        child.append(parentTemp)
    
    prepareRMQ()
    
    print(distance(10, 23))

    
    
    
    
    
  #  eulerTour = EulerTour(0, fc, ind)