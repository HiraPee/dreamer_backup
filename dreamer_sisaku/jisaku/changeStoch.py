import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import pdb
import random #現在(8/20)はランダムで変更を行う



def change(logit,rssm,stoch):
  '''
  入力 stoch カテゴライズ分布 0 or 1
  shape [n,m,32,32]
  for文減らす,1を増やしたり色々いじる


  '''
  #pdb.set_trace()
  stats = {'logit':logit}
  dist = rssm.get_dist(stats)
  #new_stoch = dist.sample()
  stoch_1 = tf.cast(dist.sample(),tf.int32)
  stoch_2 = tf.cast(dist.sample(),tf.int32)
  stoch_tof_1_2 = stoch_1 != stoch_2
  stoch_tof_1_2 = tf.cast(stoch_tof_1_2,tf.float32)
  new_stoch = stoch_tof_1_2

  return new_stoch

'''
num_stoch = stoch.numpy()
  for i in range(num_stoch.shape[0]):
    for j in range(num_stoch.shape[1]):
      for k in range(num_stoch.shape[2]):
        #pdb.set_trace()
        n = random.randrange(len(stoch[0][0][0]))
        flag = random.randrange(1)
        res = [0 for _ in range(num_stoch.shape[3])]
        res[n] = 1
        if flag == 1:
          n = random.randrange(len(stoch[0][0][0]))
          res[n] = 1
        num_stoch[i][j][k] = res

  res_stoch = tf.convert_to_tensor(num_stoch,dtype=tf.float32)

  #pdb.set_trace()'''




if __name__ == '__main__' :
  n = random.randrange(32)
  res = tf.zeros([6,5,32,32], tf.float32)
  #pdb.set_trace()
  #print(n)
  stoch = change(res)
  pass
