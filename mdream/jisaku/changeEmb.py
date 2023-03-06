import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import pdb
import random


def change(embed):
  #pdb.set_trace()
  target_embed = tf.identity(embed)
  sig = tf.math.reduce_variance(target_embed,1) #pdbででバック中は通る
  dev = tf.math.sqrt(sig)
  n = random.randrange(1,10)
  n = 10
  dev *= n
  #noise = tf.random.normal(sig.shape,stddev=dev,dtype=target_embed.dtype)
  dim_1 = []
  #pdb.set_trace()
  for i in range(target_embed.shape[0]):
    dim_2 = []
    for j in range(target_embed.shape[1]):
      noise = tf.random.normal(target_embed[i][j].shape,stddev=dev[i],dtype=target_embed.dtype)
      dim_2.append(tf.math.add(target_embed[i][j],noise))
    dim_1.append(dim_2)
    #tf.add(embed[i],add)
  #pdb.set_trace()
  t1 = tf.concat(dim_1,0)
  t1 = tf.reshape(t1,[target_embed.shape[0],target_embed.shape[1],1536])


  return t1


if __name__ == '__main__' :
  test = tf.random.uniform([6,5,1536],maxval=10)
  result = change(test)
  pdb.set_trace()
