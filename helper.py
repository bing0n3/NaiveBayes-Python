"""
calculate probability
"""
import random
from mldata import ExampleSet


def creat_cv(exset):
    random.shuffle(exset)
    # Divide exset by class label
    examples_true = ExampleSet(exset.schema)
    examples_false = ExampleSet(exset.schema)
    for example in exset:
        if example.__getitem__(-1):
            examples_true.append(example)
        else:
            examples_false.append(example)
    # make stratified cross validation
    true_number_eachfold = int(round(examples_true.__len__() / 5))
    false_number_eachfold = int(round(examples_false.__len__() / 5))


    train_sets = []
    test_sets = []
    for i in range(0,5):
        test_exset = ExampleSet(exset.schema)
        train_exset = ExampleSet(exset.schema)
        for index in range(i * true_number_eachfold, min((i + 1) * true_number_eachfold, examples_true.__len__())):
            test_exset.append(examples_true[index])
        for index in range(i * false_number_eachfold, min((i + 1) * false_number_eachfold, examples_false.__len__())):
            test_exset.append(examples_false[index])
        for example in exset:
            if example in test_exset:
                pass
            else:
                train_exset.append(example)
        train_sets.append(train_exset)
        test_sets.append(test_exset)
    return train_sets, test_sets


def training_status(exset):
    labels = [ex[-1] for ex in exset]
    label_num = labels.__len__()
    pos_num = sum(labels)
    neg_num = label_num - pos_num
