#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import os
import cPickle


class NaiveBayesClassifier:
    def __init__(self, datapath):
        """The constructor of NaiveBayesClassifier, with a parameter
        indicating where should the data stored"""
        self.datapath = datapath
        if not os.path.lexists(self.datapath):
            self._training_data = {
                    'feature' : {},
                    'class' : {},
                    'feature_when_class' : {}
                    }
            self._write_to_file()
        else:
            self._read_from_file()

    def _write_to_file(self):
        f = open(self.datapath, 'wb')
        f.write(cPickle.dumps(self._training_data))
        f.close()

    def _read_from_file(self):
        f = open(self.datapath, 'rb')
        self._training_data = cPickle.loads(f.read())
        try:
            self._training_data['feature_when_class'] = self._training_data['feature_when_gender']
            self._training_data['class'] = self._training_data['gender']
        except:
            pass
        f.close()

    def train(self, feature, a_class):
        """Train p(feature), p(feature | class), p(class)"""
        f = self._training_data
        if feature in f['feature'].keys():
            f['feature'][feature]+=1
        else:
            f['feature'][feature]=1
        if a_class in f['class'].keys():
            f['class'][a_class]+=1
        else:
            f['class'][a_class]=1
        if a_class in f['feature_when_class']:
            if feature in f['feature_when_class'][a_class]:
                f['feature_when_class'][a_class][feature]+=1
            else:
                f['feature_when_class'][a_class][feature]=1
        else:
            f['feature_when_class'][a_class] = {feature : 1}
        self._write_to_file()

    def classify(self, feature, force_class_average = False):
        """Predict which class should an object with the given feature belong to
        The first parameter is the feature.
        The second parameter is called force_class_average. If it is set to True,
        p(class) will always be 1/len(class). Otherwise, it will be 
        len(class[a_class]) / sum([len(class[i]) for i in class)"""
        f = self._training_data
        final_p = {}
        for aclass in f['class'].keys():
            try:
                p_f_c = f['feature_when_class'][aclass][feature] / \
                        sum([f['feature_when_class'][aclass][af] for af in f['feature_when_class'][aclass]])
            except KeyError:
                p_f_c = 0
            p_c = [f['class'][aclass] / sum([f['class'][c] for c in f['class']]) \
                    , 1 / len(f['class'].keys())][force_class_average]
            try:
                p_f = f['feature'][feature] / sum([f['feature'][af] for af in f['feature']])
            except KeyError:
                p_f = 0
            if p_f == 0:
                p = 0
            else:
                p = p_f_c * p_c / p_f
            final_p[aclass] = p
        if force_class_average:
            asum = 0
            for i in final_p:
                asum += final_p[i]
            for i in final_p:
                final_p[i] /= asum
        if len(final_p) >= 2:
            ans = reduce(lambda x, y: (x[1] > y[1]) and x or y, \
                map(lambda t: (t, final_p[t]), final_p) \
                )
        elif len(final_p) == 1:
            ans = final_p.keys()[0]
        else:
            ans = None
        return (ans, final_p)

