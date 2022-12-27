import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import pdb
import random


def change(logit):
  '''
  入力 deter
  shape [n,m,32,32]
  9/1 sigのルートをとってstddevに変換,さらにそれをランダムな倍数で色々試す
  '''
  #pdb.set_trace()
  #target_logit = tf.identity(logit)
  target_logit = logit
  sig = tf.math.reduce_variance(target_logit,1) #pdbででバック中は通る
  dev = tf.math.sqrt(sig)
  n = random.randrange(1,10)
  dev *= n
  #noise = tf.random.normal(sig.shape,stddev=dev,dtype=target_embed.dtype)
  dim_1 = []
  #pdb.set_trace()
  for i in range(target_logit.shape[0]):
    dim_2 = []
    for j in range(target_logit.shape[1]):
      noise = tf.random.normal(target_logit[i][j].shape,stddev=dev[i],dtype=target_logit.dtype)
      dim_2.append(tf.math.add(target_logit[i][j],noise))
    dim_1.append(dim_2)
    #tf.add(embed[i],add)
  #pdb.set_trace()
  t1 = tf.concat(dim_1,0)
  t1 = tf.reshape(t1,[target_logit.shape[0],target_logit.shape[1],target_logit.shape[2],target_logit.shape[3]])


  return t1


if __name__ == '__main__' : #test
  pass
