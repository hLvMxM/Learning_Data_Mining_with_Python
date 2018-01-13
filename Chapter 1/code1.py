#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:13:35 2018
预测 OneR算法
@author: hlvmxm
"""

# =============================================================================
# sepal lenght、sepal width、 petal length、 petal width
# 萼片 长 宽                   花瓣  长 宽
# =============================================================================

import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target
n_sample, n_features = X.shape

attribute_means = X.mean(axis=0)
X_d = np.array(X >= attribute_means, dtype='int')


from collections import defaultdict
from operator import itemgetter

# =============================================================================
# 对每一个属性进行匹配，返回最常见的类和错误数量
# =============================================================================
    
def train_feature_value(X, y_true, feature_index, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    incorrect_predictions = [class_count for class_value, class_count in class_counts.items() if class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class,error

# =============================================================================
# 返回预测器和总错误率
# =============================================================================

def train_on_feature(X, y_true, feature_index):
    values = set(X[:,feature_index])
    print(feature_index)
    predictors = {}
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true,feature_index,current_value)
        print(most_frequent_class,error)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error


# =============================================================================
# 测试
# =============================================================================

# =============================================================================
# 分割25%为测试集，75%为训练集
# =============================================================================

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=14)

#print("There are {} training samples".format(y_train.shape))
#print("There are {} testing samples".format(y_test.shape))

# =============================================================================
# 训练测试集，找到最佳匹配的类型
# =============================================================================

all_predictors = {variable: train_on_feature(X_train, y_train, variable) for variable in range(X_train.shape[1])}
errors = {variable: error for variable, (mapping,error) in all_predictors.items()}


best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
model = {'variable': best_variable, 'predictor': all_predictors[best_variable][0]}
#print(model)

def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

y_predicted = predict(X_test,model)
accuracy = np.mean(y_predicted == y_test) * 100
#print("The test accuracy is {:.1f}%".format(accuracy))

from sklearn.metrics import classification_report
#print(classification_report(y_test,y_predicted))




