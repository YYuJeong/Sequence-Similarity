# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:30 2019

@author: YuJeong
"""

from anytree import Node, RenderTree, findall, util

str1 = "aem"
str2 = "adc"
str1Len = len(str1)
str2Len = len(str2)

def GenerateItemHierarchyTree():
    item_hierarchy_tree = []
    root = Node("R")
    item_hierarchy_tree.append(root) 
    C1 = Node("C1", parent=root)
    C2 = Node("C2", parent=root)
    C3 = Node("C3", parent=root)
    C4 = Node("C4", parent=C1)
    C5 = Node("C5", parent=C1)
    g = Node("g", parent=C2)
    h = Node("h", parent=C2)
    C6 = Node("C6", parent=C3)
    C7 = Node("C7", parent=C3)
    a = Node("a", parent=C4)
    b = Node("b", parent=C4)
    C8 = Node("C8", parent=C5)
    f = Node("f", parent=C5)
    i = Node("i", parent=C6)
    j = Node("j", parent=C6)
    k = Node("k", parent=C7)
    C9 = Node("C9", parent=C7)
    c = Node("c", parent=C8)
    C10 = Node("C10", parent=C8)
    C11 = Node("C11", parent=C9)
    n = Node("n", parent=C9)
    d = Node("d", parent=C10)
    e = Node("e", parent=C10)
    l = Node("l", parent=C11)
    m = Node("m", parent=C11)
    item_hierarchy_tree.append(C1)
    item_hierarchy_tree.append(C2)
    item_hierarchy_tree.append(C3)
    item_hierarchy_tree.append(C4)
    item_hierarchy_tree.append(C5)
    item_hierarchy_tree.append(C6)
    item_hierarchy_tree.append(C7)
    item_hierarchy_tree.append(C8)
    item_hierarchy_tree.append(C9)
    item_hierarchy_tree.append(C10)
    item_hierarchy_tree.append(a)
    item_hierarchy_tree.append(b)
    item_hierarchy_tree.append(c)
    item_hierarchy_tree.append(d)
    item_hierarchy_tree.append(e)
    item_hierarchy_tree.append(f)
    item_hierarchy_tree.append(g)
    item_hierarchy_tree.append(h)
    item_hierarchy_tree.append(i)
    item_hierarchy_tree.append(j)
    item_hierarchy_tree.append(k)
    item_hierarchy_tree.append(l)
    item_hierarchy_tree.append(m)
    item_hierarchy_tree.append(n)
    return item_hierarchy_tree, root

def PrintItemHierarchyTree(itmeHierarchyTree, root):
    print("=="*30)
    for row in RenderTree(root):
        pre, fill, node = row
        print(f"{pre}{node.name}")
    print("=="*30)

def LevenshteinDistance(str1, str2, str1Len , str2Len): #Recursive
    if str1Len == 0: 
        return str2Len 
    if str2Len == 0: 
        return str1Len 

    if str1[str1Len-1]==str2[str2Len-1]: 
        cost = 0
    else:
        cost = 1    

    return min(LevenshteinDistance(str1, str2, str1Len, str2Len-1) + 1,    # Insert 
                   LevenshteinDistance(str1, str2, str1Len-1, str2Len) + 1,    # Remove 
                   LevenshteinDistance(str1, str2, str1Len-1, str2Len-1) + cost)    # Replace 

def editDistDP(str1, str2):  #Dynamic Programming 
    str1Len = len(str1)
    str2Len = len(str2)
    matrix = [[0 for x in range(str2Len + 1)] for x in range(str1Len + 1)] 
    for i in range(str1Len + 1): 
        for j in range(str2Len + 1): 
            if i == 0: 
                matrix[i][j] = j    # Min. operations = j 
            elif j == 0: 
                matrix[i][j] = i    # Min. operations = i 
            elif str1[i-1] == str2[j-1]: 
                matrix[i][j] = matrix[i-1][j-1] 
            else: 
                matrix[i][j] = 1 + min(matrix[i][j-1],        # Insert 
                                   matrix[i-1][j],        # Remove 
                                   matrix[i-1][j-1])    # Replace   
    return matrix[str1Len][str2Len], matrix 

def NewLevenshteinDistance(str1, str2, itmeHierarchyTree):
    str1Len = len(str1)
    str2Len = len(str2)
    matrix = [[0 for x in range(str2Len + 1)] for x in range(str1Len + 1)]   
    for i in range(str1Len + 1): 
        for j in range(str2Len + 1): 
            if i == 0: 
                matrix[i][j] = j    
            elif j == 0: 
                matrix[i][j] = i    
            else: # Add Hierarchy Tree 
                if str1[i-1]==str2[j-1]: 
                    cost = 0
                else:
                    cost = ComputeDiagonalCost(matrix, i, j, itmeHierarchyTree)   
                matrix[i][j] = min(matrix[i][j-1] + 1,        # Insert 
                                   matrix[i-1][j] + 1,        # Remove 
                                   matrix[i-1][j-1] + cost)    # Replace   
    return matrix[str1Len][str2Len], matrix 

def ComputeDiagonalCost(matrix, i, j, itmeHierarchyTree):

    return cost

def SearchLongestPath(item_hierarchy_tree):
    toString = list()
    for e in item_hierarchy_tree:
        toString.append(str(e))
    return toString

def ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2):
    maxlen = max(len(str1), len(str2))
    similarity = 1 - LevenshteinDist/maxlen
    return round(similarity, 3)

def PrintMatrix(matrix, str1Len, str2Len):
    print("    -", end=" ")
    for i in range(str2Len):
        print(str2[i], end=" ")
    print(" ")
    for i in range(str1Len + 1):
        if i > 0:
            print(str1[i-1], end=" ")
        else:
            print("-", end=" ")
        print("[", end=" ")
        for j in range(str2Len + 1):
            print(matrix[i][j], end=" ")
        print("]")
    print("")

if __name__ == '__main__':
    itmeHierarchyTree, root = GenerateItemHierarchyTree()
    print("itmeHierarchyTree: ")
    print(itmeHierarchyTree)
    print("root", root)
    PrintItemHierarchyTree(itmeHierarchyTree, root)
    LevenshteinDist, matrix = editDistDP(str1, str2)
    LevenshteinSim = ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2)
    print("< Original Distance Measure >")
    PrintMatrix(matrix, str1Len, str2Len)
    print("LevenshteinDistance: ", LevenshteinDist)
    print("LevenshteinSimilarity: ", LevenshteinSim)

    print("=="*30)
    print("< New Distance Measure >")
    toString = SearchLongestPath(item_hierarchy_tree)
    




















    