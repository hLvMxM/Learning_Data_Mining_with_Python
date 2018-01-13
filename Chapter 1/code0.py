# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:45:15 2018
亲和性分析
@author: 30837
"""

# =============================================================================
# （面包，牛奶，奶酪，苹果，香蕉）
# =============================================================================

# =============================================================================
# 读取txt文件affinity_dataset.txt，并且转成数组形式
# =============================================================================
import numpy as np
dataset_filename = "affinity_dataset.txt"
X = np.loadtxt(dataset_filename)
#print(X[:5])

# =============================================================================
# 统计购买苹果的顾客的数量
# =============================================================================

num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:
        num_apple_purchases += 1
#print("{0} people bought Apples".format(num_apple_purchases))

# =============================================================================
# 特别说明变量
# =============================================================================

n_samples,n_features = X.shape
features = ["bread","milk","cheese","apples","bananas"]

# =============================================================================
# valid_rules     买了一个同时买了另一个
# invalid_rules   买了一个却没有买另一个
# num_occurances  购买的个体的数量
# =============================================================================

from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)

# =============================================================================
# 统计个体与个体之间的相关性
# =============================================================================

for sample in X:
    for premise in range(5):
        if sample[premise]==0: continue
        num_occurances[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:continue
            if sample[conclusion] == 1:
                valid_rules[(premise,conclusion)] += 1
            else:
                invalid_rules[(premise,conclusion)] += 1

support = valid_rules

# =============================================================================
# 记录置信度
# =============================================================================

confidence = defaultdict(float)
for premise,conclusion in valid_rules.keys():
    rule = (premise,conclusion)
    confidence[rule] = valid_rules[rule] / num_occurances[premise]

# =============================================================================
# 打印支持度和置信度
# =============================================================================

def print_rule(premise,conclusion,support,confidence,features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name,conclusion_name))
    print(" - Support: {0}".format(support[premise,conclusion]))
    print(" - Confidence: {0:.3f}".format(confidence[(premise,conclusion)]))
        
# =============================================================================
# 支持度排序，由高到低
# =============================================================================
from operator import itemgetter
sorted_support = sorted(support.items(),key=itemgetter(1),reverse=True)

# =============================================================================
# 按照支持度排序输出
# =============================================================================

for index in range(5):
    print("Rule #{0}".format(index+1))
    premise,conclusion = sorted_support[index][0]
    print_rule(premise,conclusion,support,confidence,features)

# =============================================================================
# 置信度排序，由高到低
# =============================================================================
    
sorted_confidence = sorted(confidence.items(),key=itemgetter(1),reverse=True)

        
for index in range(5):
    print("Rule #{0}".format(index+1))
    premise,conclusion = sorted_confidence[index][0]
    print_rule(premise,conclusion,support,confidence,features)
        
        
        
        
        
        