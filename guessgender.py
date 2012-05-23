#!/usr/bin/env python
# -*- coding: utf-8 -*-



from naivebayesclassifier import NaiveBayesClassifier
from cmd import Cmd

l = NaiveBayesClassifier('data')

while True:
    name = raw_input()
    name_unicode = name.decode('utf-8')
    final_p = l.classify(name_unicode[-1], force_class_average = True)
    best_p = 0
    best_ans = -1
    for i in final_p:
        if final_p[i] > best_p:
            best_p = final_p[i]
            best_ans = i
    print status[best_ans], best_p
    if name == 'exit':
        break
