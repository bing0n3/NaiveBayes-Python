from helper import *
from mldata import *
from math import inf
import numpy as np


def nbayes(filename, cv_or_full, num_bins, m):
    examples = parse_c45(filename, '..')
    attributes = [attr for attr in examples.schema.features]
    bins = n_bin(examples, attributes, num_bins)
    if cv_or_full == 1:
        exp_list = []
        exp_list.append(examples)
        metric_average(1, exp_list, exp_list, attributes, m, bins,0)
    elif cv_or_full == 0:
        train_sets, test_sets = creat_cv(examples)
        metric_average(len(train_sets),train_sets,test_sets,attributes,m,bins,1)
        # for i in range(0,len(train_sets)):
        #     metric(train_sets[i],test_sets[i],attributes,m,bins)


    else:
        raise ValueError('option2 need be 1/0')



def nb_classify(label_probs_set, prior_probs, bins, test_example):
    probs = {}
    for label, label_probs in label_probs_set.items():
        prob = 0.0
        for i in range(1, len(test_example) - 1):
            # iterate all attributes to get prob
            val = test_example[i]
            if i in bins.keys():
                for left, right in bins[i]:
                    if left <= val < right:
                        val = (left, right)
                        break
                if not isinstance(val, tuple):
                    raise ValueError('wrong about bin')
            prob += np.log(label_probs[i][val])
        prob += np.log(prior_probs[label])
        probs[label] = prob

    sorted_probs = sorted(probs.items(), key=lambda item: item[1])
    return sorted_probs[-1][0], sorted_probs[True][1]


def nb_train(examples, attributes, m,bins):
    label_probs_set = {}
    prior_probs = {}

    for label in [True, False]:
        label_probs_set[label] = cal_mle_all_attribute(examples, attributes,
                                                       label, m, bins)
        prior_probs[label] = cal_prior_prob(examples, label)

    return label_probs_set, prior_probs, bins


def cal_mle_all_attribute(examples, attributes, label_value, m, bins):

    sub_example = [
        example for example in examples if example[-1] == label_value
    ]
    # print(sub_example)
    # len_attr = len(attributes) - 2
    # print(len(sub_example))
    pXy = {}
    for i, attribute in enumerate(attributes):
        if i == 0 or i == len(attributes) - 1:
            continue
        if i not in bins.keys():
            pXy[i] = {}
            values = attribute.values
            # hold binary, binary attribute.value equal to zero
            if attribute.type == 'BINARY':
                values = [True, False]
            len_attr = len(values)
            for attr_val in values:
                # remove missing value
                a_examples = remove_null(sub_example, i)
                pXy[i][attr_val] = cal_mle(a_examples, i, attr_val, len_attr,
                                           m)
        else:
            pXy[i] = {}
            len_attr = len(bins[i])
            for split in bins[i]:
                pXy[i][split] = cal_mle_bins(sub_example, i, split[0],
                                             split[1], len_attr, m)

    return pXy


def cal_mle(examples, attr_index, attr_value, len_attr, m):
    len_examples = len(examples)
    attr_val_count = count_attr_val(examples, attr_index, attr_value)

    if m == 0:
        # max likelihood estimate
        if len_examples == 0:
            return 0
        return cal_probs(len_examples, attr_val_count)
    else:
        # m-estimate
        if len_examples == 0:
            return 0
        return m_estimate(len_examples, attr_val_count, len_attr, m)


def cal_mle_bins(examples, attr_index, left_split, right_split, len_attr, m):
    len_examples = len(examples)
    attr_val_count = count_attr_val_split(examples, attr_index, left_split,
                                          right_split)
    if m == 0:
        if len_examples == 0:
            return 0
        return cal_probs(len_examples, attr_val_count)
    else:
        if len_examples == 0:
            return 0
        return m_estimate(len_examples, attr_val_count, len_attr, m)


def cal_probs(len_examples, len_sub):
    prob = 0.0
    prob = float(len_sub) / len_examples
    return prob


def m_estimate(len_examples, len_sub, attr_count, m):
    m_e = 0.0
    v = attr_count
    if m < 0:
        m = v
    p = 1 / float(v)
    m_e = (float(len_sub + m * p)) / (len_examples + m)
    return m_e


def cal_prior_prob(examples, label):
    count_label = count_attr_val(examples, len(examples[0]) - 1, label)
    return cal_probs(len(examples), count_label)


def group(examples, attr_index, num_bin):
    div_vals = []

    max_val = max([example[attr_index] for example in examples])
    min_val = min([example[attr_index] for example in examples])
    dis = (max_val - min_val) / num_bin
    for i in range(0, num_bin):
        if i == 0:
            div_vals.append((-inf,min_val+(i+1)*dis))
        elif i != num_bin -1:
            div_vals.append((min_val + i * dis, min_val + (i+1) * dis))
        else:
            div_vals.append((min_val + i * dis, inf))

    return div_vals


def n_bin(examples, attributes, num_bin):
    continuous_attrs = [
        i for i in range(len(attributes)) if attributes[i].type == "CONTINUOUS"
    ]
    bins = {}
    for index in continuous_attrs:
        bins[index] = group(examples, index, num_bin)
    return bins


def count_attr_val(examples, attr_index, attr_val):
    attr_val_count = 0
    for example in examples:
        if example[attr_index] == attr_val:
            attr_val_count += 1
    return attr_val_count


def count_attr_val_split(examples, attr_index, left_split, right_split):
    attr_val_count = 0
    for example in examples:
        if left_split <= example[attr_index] < right_split:
            attr_val_count += 1
    return attr_val_count


def remove_null(examples, attr_index):
    a_examples = examples.copy()
    if len(examples) == 0: return examples
    a = len(a_examples)
    i = 0
    while i < a:
        if a_examples[i][attr_index] == None:
            del (a_examples[i])
            a -= 1
            continue
        i += 1
    return a_examples

def metric_average(num_set,test_sets,train_sets,attributes,m,bins, flag):
    average_list, precision_list, recall_list, roc_list = [],[],[],[]
    mean_average = 0.0
    mean_precision = 0.0
    mean_recall = 0.0
    mean_roc = 0.0
    std_var_average, std_var_precision, std_var_recall,std_var_roc = 0,0,0,0
    for i in range(0,num_set):
        average, precision, recall, roc = metric(train_sets[i],test_sets[i],attributes,m,bins)
        average_list.append(average)
        precision_list.append(precision)
        recall_list.append(recall)
        roc_list.append(roc)
        mean_average += average
        mean_precision += precision
        mean_recall += recall
        mean_roc += roc

    mean_average /= num_set
    mean_precision /= num_set
    mean_recall /= num_set
    mean_roc /= num_set

    std_var_average= np.std(average_list)
    std_var_precision = np.std(precision_list)
    std_var_recall = np.std(recall_list)
    # std_var_roc = np.std(roc_list)

    if flag == 1:
        for i in range(0, num_set):
            print(f'{i+1}-fold:')
            print(f'Average: {round(average_list[i],3):.3f}')
            print(f'Precision: {round(precision_list[i],3):.3f}')
            print(f'Recall: {round(recall_list[i],3):.3f}')
            print(f'Area under ROC: {round(roc_list[i],3):.3f}\n\n')
        print('Average CV:')

    print(f'Average: {round(mean_average,3):.3f} {round(std_var_average, 3):.3f}')
    print(f'Precision: {round(mean_precision,3):.3f} {round(std_var_precision, 3):.3f}')
    print(f'Recall: {round(mean_recall,3):.3f} {round(std_var_recall, 3):.3f}')
    print(f'Area under ROC: {round(mean_roc,3):.3f}')



def metric(examples,test_examples, attributes,m,bins):
    label_probs_set, prior_probs,bins = nb_train(examples,attributes,m,bins)
    # print(label_probs_set)
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    tpr = [0]
    fpr = [0]
    result_set = []
    neg_all = len(examples) - 1
    for example in test_examples:
        res, true_pro = nb_classify(label_probs_set, prior_probs, bins, example)
        result_set.append((res,example[-1],true_pro))

    # sort
    result_set.sort(key=lambda x: x[-1],reverse=True)
    all_pos = len([result for result in result_set if result[1]])
    all_neg = len([result for result in result_set if not result[1]])

    for i, result in enumerate(result_set):
        if result[0]:
            if result[1]:
                tp += 1
            else:
                fp += 1
        else:
            if result[1]:
                fn += 1
            else:
                tn += 1
        tpr.append(tp/all_pos)
        fpr.append((i+1-tp)/all_neg)

    return get_accuracy(tp,fp,tn,fn),get_precision(tp,fp),get_recall(tp,fn),get_auc(tpr,fpr)

def get_accuracy(tp,fp,tn,fn):
    return (tp + tn)/(tp+fn + tn+fp)

def get_precision(tp,fp):
    # if tp+fp == 0:
    #     return 0
    return tp/(tp+fp)

def get_recall(tp,fn):
    return tp/(tp+fn)

def get_auc(tpr,fpr):
    aroc = np.trapz(fpr,tpr)
    return aroc


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise ValueError('must need 4 input')
    filename = sys.argv[1]
    cv_or_not = int(sys.argv[2])
    num_bins = int(sys.argv[3])
    m = int(sys.argv[4])

    bayes = nbayes(filename, cv_or_not, num_bins, m)
