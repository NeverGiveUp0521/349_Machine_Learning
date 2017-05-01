import numpy as np
from pylab import *
import ID3
import parse
import random


def test_result(inFile):
    number = arange(10, 310, 5)
    result_np = []
    result_p = []
    data = parse.parse(inFile)
    for j in number:
        withPruning = []
        withoutPruning = []
        for i in range(100):
            random.shuffle(data)
            train, test = data[:j], data[j:]
            tree = ID3.ID3(train, 'democrat')
            acc_test = ID3.test(tree, test)
            # print "test accuracy: ", acc
            ID3.prune(tree, test)
            acc_test_pruned = ID3.test(tree, test)
            # print "pruned tree train accuracy: ", acc
            withPruning.append(acc_test_pruned)
            withoutPruning.append(acc_test)
        average_acc_p = sum(withPruning) / len(withPruning)
        average_acc_np = sum(withoutPruning) / len(withoutPruning)
        result_np.append(average_acc_np)
        result_p.append(average_acc_p)
    print result_np
    print result_p
    print number
    xlim(0, 310)
    ylim(0.8, 1)
    xlabel('Number of training data')
    ylabel('Accuracy')
    plot(number, result_np, color='blue', label="Without pruning")
    plot(number, result_p, color='red', label="With pruning")
    legend(loc='lower right')
    show()

#test_result("house_votes_84.data")