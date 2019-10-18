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
from anytree import Node, RenderTree, findall, util, find
import time
import numpy as np
import sys
import statistics
import random

#root == 1

################################################# Taxonomy 생성  ###########################################################

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
    
    #for i in range(len(treeItem['Name'])):
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
################################################################# 오일러 투어 생성 ####################################################

global no2serial
global serial2no
global locInTrip
global depth
global k
global rangeMin
global root


k =  13988
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

def ExtractChild(parent):
    child = []
    for i in range(1, n+1):
        parentTemp = []
        for j in range(len(parent)):
            if i-1 == parent[j]:
                parentTemp.append(j+1)
        child.append(parentTemp)
    return child

##################################################### DTW  ####################################################

def NewComputeItemPath(item1, item2, maxlength):
   # start_time = time.time()
    if item1 != item2:
        itempath = distance(item1, item2)
        cost = round(itempath/maxlength, 3)
    else:
        cost = 0
   # print("---New %s seconds ---" %(time.time() - start_time))
    return cost
    


def SegDTW(str1, str2, maxlength, window=sys.maxsize):

    A, B = str1, str2
    M, N = len(A), len(B)

    cost = sys.maxsize* np.ones((M, N))

    cost[0, 0] =  NewComputeItemPath(str1[0], str2[0], maxlength)
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + NewComputeItemPath(str1[i], str2[0], maxlength)
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + NewComputeItemPath(str1[0], str2[j], maxlength)

    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) +  NewComputeItemPath(str1[i], str2[j], maxlength)

    return cost, cost[-1, -1]
    

def PrintMatrixDTW(matrix, str1, str2):
    str1Len = len(str1)
    str2Len = len(str2)
    for i in range(str2Len):
        print("    ", '{:1d}'.format(str2[i]), end=" ")
    print(" ")
    for i in range(str1Len):
        print(str1[i], end=" ")
        print("[", end=" ")
        for j in range(str2Len):
          print('{:5s}'.format(str(matrix[i][j])), end=" ")
        print("]")
    print("")
    
    
    
def SearchLongestPath(root):
    toContent = list()
    for ee in root.leaves:
        toContent.append(str(ee))
    toString = list()
    for ee in toContent:
        toString.append(ee[6:-2])
    eachNode = list()
    for ee in toString:
        eachNode.append(ee.split('/'))
    longestPath = list()
    for ee in eachNode:
        longestPath.append(ee[:-1])
    dupliPath = list(set([tuple(set(item)) for item in longestPath]))
    pathLen = list()
    for ee in dupliPath:
        pathLen.append(len(ee)-1)
    pathLen.sort(reverse=True)
    maxlength = pathLen[0] + pathLen[1]
    return maxlength

  
####################### New Dynamic Time Warping #################################
def ExtractItemPath(item1, item2, maxlength):
    str1path = str(find(root, lambda node: node.name == item1))
    str2path = str(find(root, lambda node: node.name == item2))
    str11 = str1path.split('\'')[1]
    str22 = str2path.split('\'')[1]

    str1lst = str11.split('/')
    str2lst = str22.split('/')
    
    item1parents = []
    item2parents = []
    for i in range((len(str1lst)-1)):
        item1parents.append(str1lst[i+1])
    for i in range((len(str2lst)-1)):
        item2parents.append(str2lst[i+1])
        
    return item1parents, item2parents

def ComputeItemPath(item1, item2, maxlength):
  #  start_time = time.time()
    cmpindex = 0
    if item1 != item2:
        item1parents, item2parents = ExtractItemPath(item1, item2, maxlength)
        for l in range(min(len(item1parents), len(item2parents))):
            if (l+1) == min(len(item1parents), len(item2parents)):         
                itempath = max(len(item1parents), len(item2parents))-(l+1)
            if item1parents[l] != item2parents[l]:
                cmpindex = l
                itempath = (len(item1parents)-cmpindex)+(len(item2parents)-cmpindex)
                break;
        cost = round(itempath/maxlength, 3)
    else:
        cost = 0
  #  print("---Old %s seconds ---" %(time.time() - start_time))
    return cost


def DTW_main(str1, str2, root):
    maxlength = SearchLongestPath(root) #LongestPath 계산
    start_time1 = time.time()
    matrix, cost = OriginDTW(str1, str2,maxlength, window = sys.maxsize )
    end_time1 = time.time()

    start_time2 = time.time()    
    matrix, cost = SegDTW(str1, str2,maxlength, window  = sys.maxsize)
    end_time2 = time.time()
    
    oldTime = round(end_time1 - start_time1, 4)
    newTime = round(end_time2 - start_time2, 4)
    return oldTime, newTime

def DTW1_main(str1, str2, root):
    maxlength = SearchLongestPath(root) #LongestPath 계산
    start_time1 = time.time()
    matrix, cost = OriginDTW(str1, str2,maxlength, window = 6 )
    end_time1 = time.time()

    start_time2 = time.time()    
    matrix, cost = SegDTW(str1, str2,maxlength, window  = 6)
    end_time2 = time.time()
    
    oldTime = round(end_time1 - start_time1, 4)
    newTime = round(end_time2 - start_time2, 4)
    return oldTime, newTime


def OriginDTW(str1, str2,maxlength, window=sys.maxsize):
    A, B = str1, str2
    M, N = len(A), len(B)

    cost = sys.maxsize* np.ones((M, N))

    cost[0, 0] =  ComputeItemPath(str(str1[0]), str(str2[0]), maxlength)
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + ComputeItemPath(str(str1[i]), str(str2[0]), maxlength)
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + ComputeItemPath(str(str1[0]), str(str2[j]), maxlength)

    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) +  ComputeItemPath(str(str1[i]), str(str2[j]), maxlength)

    return cost, cost[-1, -1]


####################### Generate Randeom Sequence #################################
    
def ExtractLeaf(parent):
    notleaf = parent
    notleaf.sort()
    notleaf = list(set(notleaf))
    leavess = [-1 for i in range(k)]
    for i in range(len(notleaf)):
        leavess[notleaf[i]] = 0
    leaf = []
    for i in range(len(leavess)):
        if leavess[i] == -1:
            leaf.append(i)
    return leaf

def generateRandomList(n, leaf):
    randnum = []
    for i in range(n):
        randnum.append(random.choice(leaf))
    return randnum

def generateRandomSequence():
    leaf = ExtractLeaf(parent)
    print("leaf num: ", len(leaf))
    randseq1, randseq2 = [],[]
    for mm in range(15, 25):
        for i in range(10):
            randseq1.append(generateRandomList(mm, leaf))
            randseq2.append(generateRandomList(mm, leaf))      
    random.shuffle(randseq1)
    random.shuffle(randseq2)
    return randseq1, randseq2


if __name__ == '__main__':

    treeItem = ReadCSV('Experiment Data/height_14.csv')
    item_hierarchy_tree = [] 
    global root
    root = Node("0", data = "All Item")
    item_hierarchy_tree.append(root) 

    root = GenerateItemHierarchyTree(treeItem)
    print("root height: " , root.height)

   # PrintItemHierarchyTree(root)
    
    
    queries = 3 #이 트리로 몇번 계산할지
    global n
    n = len(root.descendants)+1 #부모 노드 수
    parent = rootToZero(treeItem)
    
    child = ExtractChild(parent)
    prepareRMQ()

            
    randseq1, randseq2 = generateRandomSequence()
  
       
    print("< Runtime Test >")

    originTime = []
    segTime = []
    for i in range(len(randseq1)):
        print(i)
        oldTime, newTime = DTW_main(randseq1[i], randseq2[i], root)
        originTime.append(oldTime)
        segTime.append(newTime)
    
    print("origin Time: ", statistics.median(originTime))
    print("Segment Time: ", statistics.median(segTime))


