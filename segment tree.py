# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:28:13 2019

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

if __name__ == '__main__':
    
    treeItem = ReadCSV('eulerData.csv')
    item_hierarchy_tree = [] 

    root = Node("0", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)
    
    eulerTour = []
    par = []
    fc = [-1] * 10
    ind = 0
    eulerTour = EulerTour(0)

    for i in eulerTour:
        print("euler 경로: ", i)
