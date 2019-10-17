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


<<<<<<< HEAD
############################# 오일러 투어 생성  ###################################
'''
def EulerTour(treeIndex):

    if treeIndex == root:
        eulerTour.append(int(item_hierarchy_tree[0].name))
        childs = list(item_hierarchy_tree[0].children)

        for i in childs:
            nextIndex = int(i.name)
            EulerTour(nextIndex)
        print("#"*30)
  
    else:

        print("Not ROot: " , treeIndex)
        eulerTour.append(item_hierarchy_tree[treeIndex].name)
        print("treeInex: 추가", item_hierarchy_tree[treeIndex].name)

        parent = int(item_hierarchy_tree[treeIndex].parent.name)
        childs = list(item_hierarchy_tree[treeIndex].children)
        if item_hierarchy_tree[treeIndex].is_leaf :            
            print("지금 내 인덱스" , treeIndex, "나의 부모 노드: " , parent)
            childNum = len(list(item_hierarchy_tree[parent].children))
            childTemp = list(item_hierarchy_tree[parent].children)
            childArr = []
            for i in childTemp:
                childArr.append(i.name)
                
            for i in childArr:
                print("childARr:", i)
        #    if treeIndex == 
            print("차일드 넘: " , childNum)
            print("마지막 자식 노드: ", childArr[childNum-1])
            if treeIndex != int(childArr[childNum-1]):
                eulerTour.append(parent) 
                print("나는 마지막 자식노드 아님", treeIndex)
            else: 
                print("나는 마지막 자식 노드: " , treeIndex )
                ancestor = list(item_hierarchy_tree[treeIndex].ancestors)
                print("ancestor", ancestor)
                for i in range(len(ancestor), 1, -1):
                    eulerTour.append(int(ancestor[i-1].name))
                if item_hierarchy_tree[parent].parent.is_root:
                    print("내 부모의 부모가 루트다 0추가")
                    eulerTour.append(0)
        for i in childs:
            nextIndex = int(i.name)
            EulerTour(nextIndex)
        print("#"*30)


    return eulerTour
'''
def EulerTour(treeIndex):

    eulerTour.append(treeIndex)
    print("트리 인뎃스: ", treeIndex)
    print("eulerTour 길이: " , len(eulerTour))
    print("오일러 투어: " , eulerTour)
    print("DDDL:" , len(eulerTour)-1 )

    #print("c첫방문:" , fc)

    childStack = []
    childStack = list(item_hierarchy_tree[treeIndex].children)
    if len(childStack)  != 0: 
        for i in childStack:
         #   print("차일드 스택: ", i.name )
            EulerTour(int(i.name))
    else:
        print("스택 널 추가: " , item_hierarchy_tree[treeIndex].ancestors)
        #eulerTour.append(item_hierarchy_tree[treeIndex].parent.name)

        ancestors = list(item_hierarchy_tree[treeIndex].ancestors)
        ancestors.reverse()
        for i in ancestors:
            print("조상: " , int(i.name))
            if len(list(item_hierarchy_tree[int(i.name)].children)) != 0:
                eulerTour.append(item_hierarchy_tree[int(i.name)].name)  
        print("조상: " ,ancestors)

        #for i a
        #eulerTour.append(item_hierarchy_tree[treeIndex].parent.name)
        
    return eulerTour 
=======
############################# 오일러 투어 생성  ############################

#def EulerTour(treeIndex):

global no2serial
global serial2no
global locInTrip
global depth
global k
global rangeMin
#n = 13
k = 10001
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
 
    '''
    print("nextSerial: ", nextSerial)
    print("here: " , here )
    print("d: ", d)
    print("trip size: ", len(trip) )


    for i in range(0, len(trip)):
        print(trip[i], end=' ')
    
    print("")
    print("*"*40)
    '''
    
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

global count   
count = 0
def RMQ(array):
    global rangeMin
    global n
    n = len(array)
 #   print("n:", n)  
    rangeMin = []
    rangeMin = [0 for i in range(n*4)]   
  #  print("rangeMin size: ", len(rangeMin))
    init(array, 1, 0 , n-1)

def init(array, node, left, right):
    '''
    global count
    count = count + 1
    print("count : " , count )
    
    for i in range(0, len(array)):
        print('{:3d}'.format(array[i]), end=" ")
        if i%10 == 0:
            print("")
    print("")
        
    for i in range(0, len(rangeMin)):
        print('{:3d}'.format(rangeMin[i]), end=" ")
        if i%20 == 0:
              print("")
    print("")
    print("init node: ", node)
    print("init left: ", left)
    print("init right: ", right)
    print("*"*50)
    ''' 
    if left == right: 
       # print("좌 우 같음")
        rangeMin[node] = array[left]
       # print("if rangeMin[node]: ", rangeMin[node])
        return rangeMin[node]
    mid = int((left + right)/2)
  #  print("mid: ", mid)
    rangeMin[node] = min(init(array, node * 2 + 1, mid + 1, right), init(array, node * 2, left, mid))
    #rangeMin[node] = min(init(array, 1 * 2, 0, 12), init(array, 1 * 2 + 1, 12 + 1, 24))
   # print("min rangeMin[node]: " , rangeMin[node])
    return rangeMin[node]



def prepareRMQ():
  #  nextSerial = 0
    trip = []
    traverse(0, 0 , trip)
    return RMQ(trip)


def distance(u, v):
    #global locInTrip
    for i in range(0, len(locInTrip)):
        print("locInTrip[", i , "]: " , locInTrip[i])
    lu = locInTrip[u]
    lv = locInTrip[v]
    print("lu: "  , lu)
    print("lv: " , lv)
    if(lu > lv):
        lu, lv = lv, lu
    print("쿼리: " , query1(lu, lv))
    lca = serial2no[query1(lu, lv)]
    return depth[u]+depth[v]-2*depth[lca]


def query1(left, right):
    global n
    print("arg2 query")
    print("n: " , n)
    return query(left, right , 1, 0 , n-1)

def query(left, right, node, nodeLeft, nodeRight):
    global count
    print("*"*50)
    count = count + 1
    '''
    print("count : " , count )
    print("init node: ", node)
    print("init left: ", left)
    print("init right: ", right)
    print("nodeLeft : ", nodeLeft)
    print("nodeRight: " , nodeRight)
    '''
    if (right < nodeLeft) or (nodeRight < left):
        print("max int : ")
        return sys.maxsize
    if (left <= nodeLeft) and (nodeRight <= right):
        print("노드가 완전 포함: " , rangeMin[node])
        return rangeMin[node]
    mid = int((nodeLeft + nodeRight)/2)
    print("mid: " , mid)
    return min(query(left, right, node * 2 + 1, (mid+1), nodeRight), query(left, right, node * 2, nodeLeft, mid))
    
>>>>>>> ae066a66095e76b4790b7a738022f479d1267f2e

def rootToZero(treeItem):
    parent = []
    for i in range(len(treeItem['Parent'])):
        if treeItem['Parent'][i] == 'root':
            parent.append(0)
        else:
            parent.append(int(treeItem['Parent'][i]))
    return parent  
     
if __name__ == '__main__':
    
    treeItem = ReadCSV('C:/Users/YuJeong/Documents/Sequence-Similarity/eulerData.csv')
    item_hierarchy_tree = [] 

    root = Node("0", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
  #  PrintItemHierarchyTree(root)
    
  
    #print("후손 수:: " , len(root.descendants))
    queries = 3 #이 트리로 몇번 계산할지

    n = len(root.descendants)+1 #부모 노드 수
    parent = rootToZero(treeItem)
    #n = 13
    #parent = [ 0, 1, 1, 0 ,4]
    child = []
    for i in range(1, n+1):
        parentTemp = []
        for j in range(len(parent)):
            if i-1 == parent[j]:
                parentTemp.append(j+1)
        child.append(parentTemp)
    
    prepareRMQ()
    
<<<<<<< HEAD
    eulerTour = []
    par = []
    fc = [-1] * 10
    ind = 0
    eulerTour = EulerTour(0)

    for i in eulerTour:
        print("euler 경로: ", i)
=======
    print(distance(9523, 9459))

    
    
    
    
    
  #  eulerTour = EulerTour(0, fc, ind)
>>>>>>> ae066a66095e76b4790b7a738022f479d1267f2e
