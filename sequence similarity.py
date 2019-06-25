# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:30 2019

@author: YuJeong
"""
import csv
from anytree import Node, RenderTree, findall, util
import string
import time
import random
from math import *
import numpy as np
import sys
from collections import defaultdict


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


def cmp(a, b):
    return (a > b) - (a < b)

def GenerateItemHierarchyTree(treeItem):
    for i in range(len(treeItem['Name'])):
        globals()[treeItem['Name'][i]] =  Node(treeItem['Name'][i], parent = globals()[treeItem['Parent'][i]], data = treeItem['Data'][i])
        item_hierarchy_tree.append(globals()[treeItem['Name'][i]])
    
    ''' //입력 받아서 트리 생성
    while 1:
        nodeName, nodeData, nodeParent = input("카테고리/ 데이터/ 부모노드: ").split()
        if nodeName == "Exit":
            break
        globals()[nodeName] =  Node(nodeName, parent = globals()[nodeParent], data = nodeData)
        item_hierarchy_tree.append(globals()[nodeName])
    '''
    return root

def PrintItemHierarchyTree(root):
    print("=="*30)
    for row in RenderTree(root):
        pre, fill, node = row
        print(f"{pre}{node.name}, data: {node.data}")
    print("=="*30)

def LevenshteinDistance(str1, str2, str1LCen , str2Len): #Recursive
    str1Len = len(str1)
    str2Len = len(str2)
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
                                   matrix[i-1][j-1])   # Replace  
    return matrix[str1Len][str2Len], matrix 

def NewLevenshteinDistance(str1, str2):
    str1Len = len(str1)
    str2Len = len(str2)
    maxlength = SearchLongestPath(root) #LongestPath 계산
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
                    cost = ComputeItemPathCost(matrix,i, j, str1[i-1],str2[j-1], maxlength)
                matrix[i][j] = round(min(matrix[i][j-1] + 1,        # Insert 
                                   matrix[i-1][j] + 1,        # Remove 
                                   matrix[i-1][j-1] + cost), 3)    # Replace   
    PrintMatrix(matrix, str1, str2)
    return matrix[str1Len][str2Len]

def ComputeItemPathCost(matrix, i, j, item1, item2,  maxlength):   
    if ((matrix[i-1][j] + 1) > matrix[i-1][j-1]) and ((matrix[i][j-1] + 1) > matrix[i-1][j-1]):
        str1char = findall(root, filter_=lambda node: node.name in (item1))
        str2char = findall(root, filter_=lambda node: node.name in (item2))
        str1char = str(str1char)
        str2char = str(str2char)
        str1lst = str1char.split('/')
        str2lst = str2char.split('/')
        for l in range(min(len(str1lst), len(str2lst))):
            if str1lst[l] != str2lst[l]:
                cmpindex = l
                break
        itempath = (len(str1lst)-cmpindex)+(len(str2lst)-cmpindex)
        cost = round(itempath/maxlength, 3)
    else:
        cost = 1
    return cost

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


def ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2):
    maxlen = max(len(str1), len(str2))
    similarity = 1 - LevenshteinDist/maxlen
    return round(similarity, 3)

def PrintMatrix(matrix, str1, str2):
    str1Len = len(str1)
    str2Len = len(str2)
    print('{:5s}'.format('    -   '), end="  ")
    for i in range(str2Len):
        print('{:5s}'.format(str2[i]), end=" ")
    print(" ")
    for i in range(str1Len + 1):
        if i > 0:
            print(str1[i-1], end=" ")
        else:
            print("-", end=" ")
        print("[", end=" ")
        for j in range(str2Len + 1):
          print('{:5s}'.format(str(matrix[i][j])), end=" ")
        print("]")
    print("")
    
def PrintMatrixDTW(matrix, str1, str2):
    str1Len = len(str1)
    str2Len = len(str2)
    for i in range(str2Len):
        print("    ", '{:1s}'.format(str2[i]), end=" ")
    print(" ")
    for i in range(str1Len):
        print(str1[i], end=" ")
        print("[", end=" ")
        for j in range(str2Len):
          print('{:5s}'.format(str(matrix[i][j])), end=" ")
        print("]")
    print("")
    
def generateRandomSequence(size, chars=string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(size))

  
####################### New Dynamic Time Warping #################################
def StringToArray(str1, str2):
    stringAll = str1 + str2
    stringAll_list = list(stringAll)
    stringAll_list = list(set(stringAll_list))
    stringAll_list.sort()
    stringDic = defaultdict(lambda : None)
    for i in range(len(stringAll_list)):
        stringDic[stringAll_list[i]] = (i+1)
    str1Num, str2Num = [], []
    for i in range(len(str1)):
        str1Num.append(stringDic[str1[i]])
    for i in range(len(str2)):
        str2Num.append(stringDic[str2[i]])
        
    return  str1Num, str2Num
    
        
def NewDTW_main(str1, str2):
    matrix, cost = NewDTW(str1, str2, window = 6)
    
    PrintMatrixDTW(matrix, str1, str2)
    print('Total Distance is ', cost)


def NewDTW(str1, str2, window=sys.maxsize):
    maxlength = SearchLongestPath(root) #LongestPath 계산
    str1Num, str2Num = StringToArray(str1, str2)
    A, B = str1Num, str2Num
    M, N = len(A), len(B)


    cost = 100 * np.ones((M, N))

    cost[0, 0] =  ComputeItemPath(str1[0], str2[0], maxlength)
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + ComputeItemPath(str1[i], str2[0], maxlength)
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + ComputeItemPath(str1[0], str2[j], maxlength)

    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) +  ComputeItemPath(str1[i], str2[j], maxlength)

    return cost, cost[-1, -1]

def ComputeItemPath(item1, item2, maxlength):
    if item1 != item2:
        str1path = findall(root, filter_=lambda node: node.name in (item1))
        str2path = findall(root, filter_=lambda node: node.name in (item2))
        str1path = str(str1path)
        str2path = str(str2path)
        str1lst = str1path.split('/')
        str2lst = str2path.split('/')
        for l in range(min(len(str1lst), len(str2lst))):
            if str1lst[l] != str2lst[l]:
                cmpindex = l
                break
        itempath = (len(str1lst)-cmpindex)+(len(str2lst)-cmpindex)
        cost = round(itempath/maxlength, 3)
    else:
        cost = 0
    return cost

############################## Origin DTW ##########################################
def DTW_main(str1, str2):

    matrix, cost = DTW(str1, str2, window = 6)

    PrintMatrixDTW(matrix, str1, str2)
    print('Total Distance is ', cost)


def DTW(str1, str2, window=sys.maxsize, d=lambda x, y: abs(x - y)):
    str1Num, str2Num = StringToArray(str1, str2)
    A, B = str1Num, str2Num
    M, N = len(A), len(B)

    cost = 100 * np.ones((M, N))

    cost[0, 0] =  d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(A[i], B[0])
    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(A[0], B[j])

    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) +  d(A[i], B[j])

    return cost, cost[-1, -1]

######################### Origin Needleman-Wunsch ###################################

def zeros(rows, cols):
    retval = []
    for x in range(rows):
        retval.append([])
        for y in range(cols):
            retval[-1].append(0)
    return retval

def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == '-' or beta == '-':
        return gap_penalty
    else:
        return mismatch_penalty

def needleman_wunsch(seq1, seq2):
    n = len(seq1)
    m = len(seq2)
    score = zeros(m+1, n+1)
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + match_score(seq1[j-1], seq2[i-1])
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)
    print("NW score: ", score[m][n])
    return score

def needleman_wunsch_align(score, seq1, seq2):   
    n = len(seq1)
    m = len(seq2)
    align1 = ""
    align2 = ""
    i = m
    j = n 
    while i > 0 and j > 0: 
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]
        
        if score_current == score_diagonal + match_score(seq1[j-1], seq2[i-1]):
            align1 += seq1[j-1]
            align2 += seq2[i-1]
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1 += seq1[j-1]
            align2 += '-'
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 += '-'
            align2 += seq2[i-1]
            i -= 1

    while j > 0:
        align1 += seq1[j-1]
        align2 += '-'
        j -= 1
    while i > 0:
        align1 += '-'
        align2 += seq2[i-1]
        i -= 1
    
    align1 = align1[::-1]
    align2 = align2[::-1]
    
    return(align1, align2)

######################### New Needleman-Wunsch ###################################
def New_NW(seq1, seq2):
    
    n = len(seq1)
    m = len(seq2)
    maxlength = SearchLongestPath(root) #LongestPath 계산
    score = zeros(m+1, n+1)
    for i in range(0, m + 1):
        score[i][0] = gap_penalty * i
    for j in range(0, n + 1):
        score[0][j] = gap_penalty * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] +   New_match_score(seq1[j-1], seq2[i-1], maxlength )
            delete = score[i - 1][j] + gap_penalty
            insert = score[i][j - 1] + gap_penalty
            score[i][j] = max(match, delete, insert)
    print("NW score: ", score[m][n])
    return score

def New_match_score(alpha, beta, maxlength):
    if alpha == '-' or beta == '-':
        return gap_penalty
    else:
        return (1-ComputeItemPath(alpha, beta, maxlength))
    
def NEW_NW_align(score, seq1, seq2):   
    n = len(seq1)
    m = len(seq2)
    align1 = ""
    align2 = ""
    i = m
    j = n 
    maxlength = SearchLongestPath(root) #LongestPath 계산
    while i > 0 and j > 0: 
        score_current = score[i][j]
        score_diagonal = score[i-1][j-1]
        score_up = score[i][j-1]
        score_left = score[i-1][j]
        
        if score_current == score_diagonal + New_match_score(seq1[j-1], seq2[i-1], maxlength ):
            align1 += seq1[j-1]
            align2 += seq2[i-1]
            i -= 1
            j -= 1
        elif score_current == score_up + gap_penalty:
            align1 += seq1[j-1]
            align2 += '-'
            j -= 1
        elif score_current == score_left + gap_penalty:
            align1 += '-'
            align2 += seq2[i-1]
            i -= 1

    while j > 0:
        align1 += seq1[j-1]
        align2 += '-'
        j -= 1
    while i > 0:
        align1 += '-'
        align2 += seq2[i-1]
        i -= 1
    
    align1 = align1[::-1]
    align2 = align2[::-1]
    
    return(align1, align2)
    
if __name__ == '__main__':
    str1 = "akgaa"
    str2 = " "
    treeItem = ReadCSV('tree.csv')
   # data = ReadCSV('data.csv')
    
    item_hierarchy_tree = []    
    root = Node("R", data = "All Item")
    item_hierarchy_tree.append(root) 
    root = GenerateItemHierarchyTree(treeItem)
    PrintItemHierarchyTree(root)
    
    #maxlength = SearchLongestPath(root) #LongestPath 계산
    
    LevenshteinDist, matrix = editDistDP(str1, str2)
    LevenshteinSim = ComputeLevenshteinSimilarity(LevenshteinDist, str1, str2)
    
    print("< Original Levenshtein Measure >")
    PrintMatrix(matrix, str1, str2)
    print("LevenshteinDistance: ", LevenshteinDist)
    print("LevenshteinSimilarity: ", LevenshteinSim)
    print("=="*30)
    
    print("< New Levenshtein Measure >")
    NewLevenshteinDist = NewLevenshteinDistance(str1, str2)
   # PrintMatrix(Newmatrix, str1, str2)
    NewLevenshteinSim = ComputeLevenshteinSimilarity(NewLevenshteinDist, str1, str2)
    print("LevenshteinDistance: ", NewLevenshteinDist)
    print("LevenshteinSimilarity: ", NewLevenshteinSim)
    print("=="*30)
    
    print("< Origin Dynamic Time Warping Measure >")
    DTW_main(str1, str2)
    print("=="*30)
    
    print("< New Dynamic Time Warping Measure >")
    NewDTW_main(str1, str2)
    print("=="*30)
    
    print("< Origin Needleman-Wunsch Measure >")  
    gap_penalty = -1
    match_award = 1
    mismatch_penalty = 0
    score1 = needleman_wunsch(str1, str2)
    PrintMatrix(score1, str2, str1)
    output1, output2 = needleman_wunsch_align(score1 ,str1, str2)
    print(output1 + "\n" + output2)   
    print("=="*30)
    
    print("< New Needleman-Wunsch Measure >")  
    score1 = New_NW(str1, str2) 
    PrintMatrix(score1, str2, str1)
    output1, output2 = NEW_NW_align(score1 ,str1, str2)
    print(output1 + "\n" + output2)
    
    '''
    print("=="*30)
    print("< Runtime Test >")
    randseq1, randseq2 = [],[]
    for i in range(3, 11):
        for j in range(1250):
            randseq1.append(generateRandomSequence(i))
            randseq2.append(generateRandomSequence(i))
    random.shuffle(randseq1)
    random.shuffle(randseq2)   
    print("Sequence Num : " , len(randseq1))
    
    start_time = time.time()
    for i in range(len(randseq1)):
         NewLevenshteinDist = NewLevenshteinDistance(randseq1[i], randseq2[i])
    print("---Edit %s seconds ---" %(time.time() - start_time))
    
    start_time = time.time()
    for i in range(len(randseq1)):
        NewDTW_main(randseq1[i], randseq2[i])
    print("---DTW %s seconds ---" %(time.time() - start_time))

    start_time = time.time()
    for i in range(len(randseq1)):
        score1 = New_NW(randseq1[i], randseq2[i])
    print("---NW %s seconds ---" %(time.time() - start_time))
    print("=="*30)
    '''
    