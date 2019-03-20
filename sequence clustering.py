# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:30 2019

@author: YuJeong
"""

from anytree import Node, RenderTree

str1 = "aem"
str2 = "adc"
str1Len = len(str1)
str2Len = len(str2)

def GenerateItemHierarchyTree():
    item_hierarchy_tree = []
    root = Node("R")
    item_hierarchy_tree.append(root) 
    for i in range(0,3):
        new_node = Node(f'C{i+1}', parent=root)
        item_hierarchy_tree.append(new_node)
    item_hierarchy_tree.append(Node("C4", parent=root.children[0]))
    item_hierarchy_tree.append(Node("a", parent=root.children[0].children[0]))
    item_hierarchy_tree.append(Node("b", parent=root.children[0].children[0]))
    item_hierarchy_tree.append(Node("C5", parent=root.children[0]))
    item_hierarchy_tree.append(Node("C8", parent=root.children[0].children[1]))
    item_hierarchy_tree.append(Node("f", parent=root.children[0].children[1]))
    item_hierarchy_tree.append(Node("c", parent= root.children[0].children[1].children[0]))
    item_hierarchy_tree.append(Node("C10", parent=root.children[0].children[1].children[0]))
    item_hierarchy_tree.append(Node("d", parent= root.children[0].children[1].children[0].children[1]))
    item_hierarchy_tree.append(Node("e", parent=root.children[0].children[1].children[0].children[1]))
    item_hierarchy_tree.append(Node("g", parent=root.children[1]))
    item_hierarchy_tree.append(Node("h", parent=root.children[1]))
    item_hierarchy_tree.append(Node("C6", parent=root.children[2]))
    item_hierarchy_tree.append(Node("C7", parent=root.children[2]))
    item_hierarchy_tree.append(Node("i", parent=root.children[2].children[0]))
    item_hierarchy_tree.append(Node("j", parent=root.children[2].children[0]))
    item_hierarchy_tree.append(Node("k", parent=root.children[2].children[1]))
    item_hierarchy_tree.append(Node("C9", parent=root.children[2].children[1]))
    item_hierarchy_tree.append(Node("C11", parent= root.children[2].children[1].children[1]))
    item_hierarchy_tree.append(Node("n", parent=root.children[2].children[1].children[1]))
    item_hierarchy_tree.append(Node("l", parent= root.children[2].children[1].children[1].children[0]))
    item_hierarchy_tree.append(Node("m", parent=root.children[2].children[1].children[1].children[0]))
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





















    