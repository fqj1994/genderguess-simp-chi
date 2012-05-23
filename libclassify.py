#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import cPickle

def train(Feature, Gender):
	"""Train p(Gender), p(Feature | Gender), p(Feature)"""
	f = open('data', 'rb')
	data = cPickle.loads(f.read())
	f.close()
	if data['gender'].has_key(Gender):
		data['gender'][Gender]+=1
	else:
		data['gender'][Gender]=1
	if data['feature'].has_key(Feature):
		data['feature'][Feature]+=1
	else:
		data['feature'][Feature]=1
	if data['feature_when_gender'].has_key(Gender):
		if data['feature_when_gender'][Gender].has_key(Feature):
			data['feature_when_gender'][Gender][Feature]+=1
		else:
			data['feature_when_gender'][Gender][Feature]=1
	else:
		data['feature_when_gender'][Gender] = { Feature : 1}
	f = open('data', 'wb')
	f.write(cPickle.dumps(data))
	f.close()

def predict(Feature):
	"""predit use Bayes Rule.
	P(Gender|Feature) = P(Feature | Gender) * P(Gender) / P(Feature)
	"""
	f = open('data', 'rb')
	data = cPickle.loads(f.read())
	f.close()
	Prefer = -1
	Prefer_P = 0
	for Gender in range(2):
		try:
			p_f_g = data['feature_when_gender'][Gender][Feature] /\
					sum([data['feature_when_gender'][Gender][f] for f in data['feature_when_gender'][Gender]])
		except KeyError:
			p_f_g = 0
		p_g = data['gender'][Gender] /\
				sum([data['gender'][g] for g in data['gender']])
		p_f = data['feature'][Feature] /\
				sum([data['feature'][f] for f in data['feature']])
		p = p_f_g * p_g / p_f
		if p_f_g > Prefer_P:
			Prefer_P = p_f_g
			Prefer = Gender
	return Prefer


def init():
	f = open('data', 'wb')
	data = {'gender' : {}, 'feature' : {}, 'feature_when_gender' : {}}
	f.write(cPickle.dumps(data))
	f.close()
