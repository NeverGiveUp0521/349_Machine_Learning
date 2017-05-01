# Group members: Zhanghexuan Ji (zjv5649),	Boyu Xu (bxp6650), 	Daowei Li (dlx4580)

from node import Node
from math import *
from random import *
from parse import *


def ID3(examples, default):
    if len(examples) == 0:
        leaf = Node()
        leaf.label = default
        leaf.isleaf = 1
        return leaf
    elif all(examples[0]['Class'] == example['Class'] for example in examples):
        leaf = Node()
        leaf.label = examples[0]['Class']
        leaf.isleaf = 1
        leaf.train_num = train_number(examples)
        return leaf
    elif not check_split(examples):
        leaf = Node()
        leaf.label = Mode(examples)
        leaf.isleaf = 1
        leaf.train_num = train_number(examples)
        return leaf
    else:
        best = best_feature(examples)
        if best == '?':
            leaf = Node()
            leaf.label = Mode(examples)
            leaf.isleaf = 1
            leaf.train_num = train_number(examples)
            return leaf
        tree = Node()
        tree.label = best
        AttrSet = list(set([example[best] for example in examples]))
        for vi in AttrSet:
            # examples_i = [example for example in examples if example[best] == vi or example[best] == '?']
            examples_i = [example for example in examples if example[best] == vi]
            subtree = ID3(examples_i, Mode(examples))
            subtree.parent = tree
            tree.children[vi] = subtree
        tree.train_num = train_num_from_child(tree)
        return tree

    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''


def test(node, examples):
    result = []  # Class list of the test data (in order)
    for example in examples:
        result.append(evaluate(node, example))
    test = [example['Class'] for example in examples]
    accunum = sum([result[i] == test[i] for i in range(len(examples))])
    total = len(examples)
    accurate = float(accunum) / total
    return accurate
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''


def evaluate(node, example):
    p = node
    while len(p.children) > 0:
        attr = p.label
        val = example[attr]
        # if val == '?':
        #     keylist = p.children.keys()
        #     valuelist = [sum(x.train_num.values()) for x in p.children.values()]
        #     imax = valuelist.index(max(valuelist))
        #     val = keylist[imax]
        if p.children.get(val, None) is not None:
            p = p.children[val]
        elif val == '?':
            keylist = p.children.keys()
            valuelist = [sum(x.train_num.values()) for x in p.children.values()]
            imax = valuelist.index(max(valuelist))
            val = keylist[imax]
            p = p.children[val]
        elif '?' in p.children.keys():
            val = '?'
            p = p.children[val]
        else:
            keylist = p.train_num.keys()
            vlist = [p.train_num[k] for k in keylist]
            imax = vlist.index(max(vlist))
            return keylist[imax]
    return p.label

    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''


def prune(node, examples):
    postlist = post_trans(node)
    # print [x.label for x in postlist]
    for pnode in postlist:
        if pnode.isleaf == 0 and pnode.parent is None:  # root node
            #print 'root node! '
            pre_acc = test(node, examples)
            #print pre_acc
            leaf = Node()
            leaf.isleaf = 1
            leaf.train_num = pnode.train_num
            keylist = pnode.train_num.keys()
            numlist = [pnode.train_num[k] for k in keylist]
            nummax = max(numlist)
            ikey = [i for i, j in enumerate(numlist) if j == nummax]
            # print ichild
            if len(ikey) > 1:
                acclist = []
                for ik in ikey:
                    leaf.label = keylist[ik]
                    acclist.append(test(leaf, examples))
                imax = acclist.index(max(acclist))
                ikey = ikey[imax]
                leaf.label = keylist[ikey]
                prune_acc = test(leaf, examples)
            # ichild = numlist.index(max(numlist))
            else:
                leaf.label = keylist[ikey[0]]
                prune_acc = test(leaf, examples)
                #print prune_acc
            if (pre_acc <= prune_acc):
                #print 'pruning'
                #print leaf.breadth_first_search()
                node.set_node(leaf)
        if pnode.isleaf == 0 and pnode.parent is not None:
            # print pnode.label
            pa = pnode.parent
            for key in pa.children.keys():
                if pnode == pa.children[key]:
                    pkey = key
                    break
            pre_acc = test(node, examples)
            leaf = Node()
            leaf.isleaf = 1
            leaf.parent = pa
            leaf.train_num = pnode.train_num
            pa.children[pkey] = leaf
            keylist = pnode.train_num.keys()
            numlist = [pnode.train_num[k] for k in keylist]
            nummax = max(numlist)
            ikey = [i for i, j in enumerate(numlist) if j == nummax]
            # print ichild
            if len(ikey) > 1:
                acclist = []
                for ik in ikey:
                    leaf.label = keylist[ik]
                    acclist.append(test(node, examples))
                imax = acclist.index(max(acclist))
                ikey = ikey[imax]
                leaf.label = keylist[ikey]
                prune_acc = test(node, examples)
            # ichild = numlist.index(max(numlist))
            else:
                leaf.label = keylist[ikey[0]]
                prune_acc = test(node, examples)
            if (pre_acc > prune_acc):
                pa.children[pkey] = pnode

    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order32w3
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''


# return true if data can still be split
def check_split(examples):
    attrlist = examples[0].keys()
    attrlist.remove('Class')
    for attr in attrlist:
        if len(set([example[attr] for example in examples])) > 1:
            return True
    return False


def dropmiss(examples):
    l = len(examples[0]) - 1
    for i, example in enumerate(examples):
        if example.values().count('?') > l/2:
            examples.pop(i)


def fillMiss(examples):
    attrlist = examples[0].keys()
    classlist = list(set([x['Class'] for x in examples]))
    dclass = {}
    for c in classlist:
        dclass[c] = [ex for ex in examples if ex['Class'] == c]
    for att in attrlist:
        vmax = {}
        for k in dclass.keys():
            dex = dclass[k]
            vmax[k] = ModeByAttr(dex, att)
        for i, example in enumerate(examples):
            if example[att] == '?':
                examples[i][att] = vmax[example['Class']]


def fillMiss_n(examples, att):
    #attrlist = examples[0].keys()
    classlist = list(set([x['Class'] for x in examples]))
    dclass = {}
    for c in classlist:
        dclass[c] = [ex for ex in examples if ex['Class'] == c]
    vmax = {}
    for k in dclass.keys():
        dex = dclass[k]
        vmax[k] = ModeByAttr(dex, att)
    for i, example in enumerate(examples):
        if example[att] == '?':
            examples[i][att] = vmax[example['Class']]


def miss_class(examples, att):
    #attrlist = examples[0].keys()
    classlist = list(set([x['Class'] for x in examples]))
    dclass = {}
    for c in classlist:
        dclass[c] = [ex for ex in examples if ex['Class'] == c]
    vmax = {}
    for k in dclass.keys():
        dex = dclass[k]
        vmax[k] = ModeByAttr(dex, att)
    return vmax
    # for i, example in enumerate(examples):
    #     if example[att] == '?':
    #         examples[i][att] = vmax[example['Class']]


def fillMisst(examples):
    attrlist = examples[0].keys()
    for att in attrlist:
        vmax = ModeByAttr(examples, att)
        for i, example in enumerate(examples):
            if example[att] == '?':
                examples[i][att] = vmax

def fillMisst_n(examples, att):
    # attrlist = examples[0].keys()
    vmax = ModeByAttr(examples, att)
    for i, example in enumerate(examples):
        if example[att] == '?':
            examples[i][att] = vmax


# get mode class value
def Mode(examples):
    CLlist = [example['Class'] for example in examples]
    return max(CLlist, key=CLlist.count)


# get mode attribute value
def ModeByAttr(examples, attr):
    CLlist = [example[attr] for example in examples if example[attr] != '?']
    return max(CLlist, key=CLlist.count)


# determine if the node should be prune
def node_prune(pretree, prunetree, examples):
    pre_acc = test(pretree, examples)
    prune_acc = test(prunetree, examples)
    return prune_acc >= pre_acc


# check if the node is available for prune checking
def prune_test(node):
    childlist = node.children.values()
    leaflist = [child.isleaf for child in childlist]
    return all(leaflist)


# post order transverse the tree, return the node list
def post_trans(node):
    traversal, stack = [], [(node, False)]
    while stack:
        node, visited = stack.pop()
        if node:
            if visited:
                # add to result if visited
                traversal.append(node)
            else:
                # post-order
                stack.append((node, True))
                keylist = node.children.keys()
                keylist.reverse()
                for key in keylist:
                    stack.append((node.children[key], False))
    return traversal


# split the data into training set and testing set
def split_examples(examples, train_num):
    random.shuffle(examples)
    training, testing = examples[:train_num], examples[train_num:]
    return training, testing


def determine(llist):
    flag = True
    reserve = llist[0].get('Class')
    for di in llist:
        if (reserve != di.get('Class')):
            flag = False
            break
    return flag


# calculate entropy
def get_prior(examples):
    clist = [example['Class'] for example in examples]
    s = list(set(clist))
    pdict = {}
    for val in s:
        num_r = clist.count(val)
        pdict[val] = float(num_r) / len(examples)
    # num_r = sum([1 if x['Class'] == s[0] else 0 for x in examples])
    return pdict


# entrophy
def calc_entropy(examples):
    if len(examples) == 0:
        return 0
    pdict = get_prior(examples)
    sum_en = 0.0
    for key in pdict.keys():
        p = pdict[key]
        if p == 0 or p == 1:
            sum_en = sum_en + 0
        else:
            sum_en = sum_en - p * log(p, 2)
    return sum_en


# get_subset
def get_subset(examples, fea_name, fea_value):
    subset = filter(lambda x: x[fea_name] == fea_value, examples)
    return subset


def train_number(examples):
    classlist = [x['Class'] for x in examples]
    classset = list(set(classlist))
    train_num = {}
    for val in classset:
        train_num[val] = classlist.count(val)
    return train_num


def train_num_from_child(node):
    train = {}
    childlist = node.children.values()
    classlist = []
    for child in childlist:
        classlist = classlist + child.train_num.keys()
    classset = list(set(classlist))
    for val in classset:
        train[val] = sum([x.train_num.get(val, 0) for x in childlist])
    return train


# calculate information gain.
def infoGain(examples, feature_name):
    examples_entropy = calc_entropy(examples)
    total = len(examples)
    Aset = list(set([example[feature_name] for example in examples]))  # value set of feature
    subset_missing = []
    # if '?' in Aset:
    #     Aset.remove('?')
    #     subset_missing = get_subset(examples, feature_name, "?")
    sublist = {}
    clist = []
    # if len(Aset) == 0:
    #     return 0
    for vi in Aset:
        sublist[vi] = get_subset(examples, feature_name, vi)
        #clist.append(len(sublist[vi]))
    # imax = clist.index(max(clist))
    # vmax = Aset[imax]  # mode value
    # sublist[vmax] = subset_missing + sublist[vmax]
    for suben in sublist.values():
        h = calc_entropy(suben)
        examples_entropy = examples_entropy - float(len(suben)) / total * h
    return examples_entropy


def best_feature(examples):
    gmax = 0
    atr = '?'
    attributes = examples[0].keys()
    attributes.remove('Class')
    # atr = attributes[0]
    for fea_name in attributes:
        if (gmax < infoGain(examples, fea_name)):
            gmax = infoGain(examples, fea_name)
            atr = fea_name
    return atr
