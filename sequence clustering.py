# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:30 2019

@author: YuJeong
"""

from anytree import Node, RenderTree

## 가능하면 아래처럼 node set list를 하나 만들어서 관리해주는 것이 편함. 
all_node_set = []

## 새로운 변수를 추가해서 넣어줘도 상관없음. 
## 단 하나의 어떤 node에 data가 있을 경우 아래 모든 노드에서도 data를 넣어주어야 함
root = Node("root", data=0)

all_node_set.append(root)

for i in range(0, 3):
    ## root.children은 기본적으로 tuple구조이며, 따라서 append등으로 새로운 값을 넣어줄 수 없음
    ## 대신 아래처럼 새로운 node를 만들고, parent를 지정해주면 알아서 연결됨 
    new_node = Node(f'child_{i}', parent=root, data=0)
    ## child가 추가되면 data를 변경하도록 세팅 
    root.data+=1
    all_node_set.append(new_node)
Node("child_child_1", parent=root.children[0], data=0)

print("=="*20)
## text상에서, tree를 예쁘게 볼 수 있음. 
for row in RenderTree(root):
    pre, fill, node = row
    print(f"{pre}{node.name}, data: {node.data}")
print("=="*20)
## 기본적인 tree method를 지원
print(f"children: { [c.name for c in root.children] }")
print(f"parent: {root.children[0].parent}")
print(f"is_root: {root.is_root}")
print(f"is_leaf: {root.is_leaf}")
## path ==> root부터 target_Node까지의 길을 말함. 
target_node = root.children[0].children[0]
print(f"path: {target_node.path}")
print(f"ancestors: {target_node.ancestors}")
print("=="*20)