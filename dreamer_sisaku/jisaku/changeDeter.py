import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import pdb
import random #現在(8/20)はランダムで変更を行う

'''
def change(deter):
  入力 deter
  shape [n,m,600]
  9/1 sigのルートをとってstddevに変換,さらにそれをランダムな倍数で色々試す
  sig = tf.math.reduce_variance(deter,1) #pdbででバック中は通る
  dev = tf.math.sqrt(sig)
  #n = random.randrange(9)
  #dev *= n
  noise = tf.random.normal(sig.shape,stddev=dev,dtype=deter.dtype)
  #pdb.set_trace()
  for i in range(deter.shape[0]):
    add = noise[i]
    for j in range(deter.shape[1]):
      tf.add(deter[i][j],add)

  #pdb.set_trace()

  return deter
'''

def change(deter):
  #pdb.set_trace()
  target_deter = tf.identity(deter)
  sig = tf.math.reduce_variance(target_deter,1) #pdbででバック中は通る
  dev = tf.math.sqrt(sig)
  n = random.randrange(10,20)
  dev *= n
  #noise = tf.random.normal(sig.shape,stddev=dev,dtype=target_embed.dtype)
  dim_1 = []
  #pdb.set_trace()
  for i in range(target_deter.shape[0]):
    dim_2 = []
    for j in range(target_deter.shape[1]):
      noise = tf.random.normal(target_deter[i][j].shape,stddev=dev[i],dtype=target_deter.dtype)
      dim_2.append(tf.math.add(target_deter[i][j],noise))
    dim_1.append(dim_2)
    #tf.add(embed[i],add)
  #pdb.set_trace()
  t1 = tf.concat(dim_1,0)
  t1 = tf.reshape(t1,[target_deter.shape[0],target_deter.shape[1],target_deter.shape[2]])


  return t1

if __name__ == '__main__' : #test
  pass
