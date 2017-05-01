# Group members: Zhanghexuan Ji (zjv5649),	Boyu Xu (bxp6650), 	Daowei Li (dlx4580)

import copy


class Node:
    def __init__(self):
        self.label = None
        self.children = {}

    # you may want to add additional fields here...
        self.isleaf = 0  # 1 if the node is a leaf node; default 0
        self.parent = None  # the parent pointer links to the parent node
        self.train_num = {}  # the data number for training of the leaf node; 0 if not leaf node

    # initiate a node with an exist node n
    def set_node(self, n):
        self.label = n.label
        self.children = n.children
        self.isleaf = n.isleaf
        self.parent = n.parent
        self.train_num = n.train_num

    # deepcopy the node/tree
    def copy(self):
        return copy.deepcopy(self)

    def get_label(self):
        return self.label
        '''
        given a node, will return the value at this node
        '''

    def get_children(self):
        return self.children
        '''
        given a node, will return the children of this node
        '''

    def get_isleaf(self):
        return self.isleaf

    def get_parent(self):
        return self.parent

    def set_label(self, label):
        self.label = label

    def set_children(self, children):
        self.children = children

    def set_isleaf(self, isleaf):
        self.isleaf = isleaf

    def set_parent(self, parent):
        self.parent = parent

    def breadth_first_search(self):
        staNode = []  # Queue stack for Node
        queVal = []  # output string of tree values
        staNode.append(self)
        while len(staNode) > 0:
            tempn = staNode.pop(0)  # Node tempn stores the Node being popped
            queVal.append(str(tempn.get_label()))  # push the value in the string
            if tempn.get_children() is not None:
                staNode.extend(tempn.get_children().values())  # push the Child Nodes in the Queue
        return ' '.join(queVal)
        '''
        given the root node, will complete a breadth-first-search on the tree, returning the value of each node in the correct order
        '''
