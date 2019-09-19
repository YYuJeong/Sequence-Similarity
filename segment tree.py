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
def EulerTour(treeIndex):
    eulerTour = []

    for i in eulerTour:
        print("euler tour : ", i)
    if treeIndex == root:
        print("root")
        eulerTour.append(item_hierarchy_tree[0].name)
        childs = item_hierarchy_tree[0].children
        childs = list(childs)
        print("childs: ", childs[0])
        print("typ:", type(childs[0]))
        for i in childs:
            print(i.name)
            nextIndex = i.name
            EulerTour(nextIndex)
  #      for i in range(0, len(childs)):
  #          print("child name:", childs)
  #          eulerTour(childs[i].name)
    else:
        print("not Root", treeIndex)
        print()
        childs = item_hierarchy_tree[treeIndex].children
        childs = list(childs)
        for i in range(0, len(childs)):
            print("child name:", childs[i].name)
        
    #if childs
    



if __name__ == '__main__':
    
    treeItem = ReadCSV('eulerData.csv')
    
    item_hierarchy_tree = [] 

    root = Node("1", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)

    EulerTour(root)