# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:30 2019

@author: YuJeong
"""

from anytree import Node, RenderTree

str1 = "aem"
str2 = "adc"
m = len(str1)
n = len(str2)

def GenerateItemHierarchyTree():
    item_hierarchy_tree = []
    root = Node("root")
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

def LevenshteinDistance(str1, str2, m , n): #Recursive
    if m==0: 
        return n 
    if n==0: 
        return m 

    if str1[m-1]==str2[n-1]: 
        cost = 0
    else:
        cost = 1    

    return min(LevenshteinDistance(str1, str2, m, n-1) + 1,    # Insert 
                   LevenshteinDistance(str1, str2, m-1, n) + 1,    # Remove 
                   LevenshteinDistance(str1, str2, m-1, n-1) + cost)    # Replace 

def editDistDP(str1, str2, m, n):  #Dynamic Programming
    # Create a table to store results of subproblems 
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m+1): 
        for j in range(n+1): 
  
            # If first string is empty, only option is to 
            # insert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n], dp 

def ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2):
    maxlen = max(len(str1), len(str2))
    similarity = 1-LevenshteinDist/maxlen
    return similarity

if __name__ == '__main__':
    itmeHierarchyTree, root = GenerateItemHierarchyTree()
    PrintItemHierarchyTree(itmeHierarchyTree, root)
    LevenshteinDist,dp = editDistDP(str1, str2, m, n)
    LevenshteinSim = ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2)

    