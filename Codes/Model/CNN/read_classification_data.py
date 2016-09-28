"""Functions for reading Seismic data."""
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy
import tensorflow as tf
from six.moves import urllib
from six.moves import xrange  
import random
import sys

def read_data(data_dir,dtype=tf.float32):
 f = open(data_dir)             
 data_list = f.readlines()
 return data_list

def random_read_next_batch(data_sample, data_label,scale, num, dim):
 sample = [[0.0 for col in range(dim)] for row in range(num)]
 label = [[0.0 for col in range(2)] for row in range(num)]
 rand = random.sample(range(scale), num)
 for m in range(num):
  if data_label[rand[m]].strip('\n') == '0':
   label[m][0] = 1.0
  else:
   label[m][1] = 1.0
  s = data_sample[rand[m]].strip('\n').split(',')
  for n in range(dim):
   sample[m][n] = s[n]
 return sample, label

def read_next_batch(data_sample, data_label, scale, begin, end, dim):
 if end > scale:
  end = scale
 sample = [[0.0 for col in range(dim)] for row in range(end - begin)]
 label = [[0.0 for col in range(2)] for row in range(end - begin)]
 for m in range(end - begin):
  if data_label[m + begin].strip('\n') == '0':
   label[m][0] = 1.0
  else:
   label[m][1] = 1.0
  s = data_sample[m + begin].strip('\n').split(',')
  for n in range(dim):
   sample[m][n] = s[n]
 return sample, label
