import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import tensorflow as tf

import pdb

#@tf.function
def add_noise(target,get_sig =False):
  #tf.config.experimental_run_functions_eagerly(True)  tf.executing_eagerly() tf.config.run_functions_eagerly(True)

  # deterの次元数が違うから途中noise付与でエラーが出る
  dim = tf.shape(target)# [6,5,32,32]
  dim_list = []
  for i in range(len(dim)):
    dim_list.append(i) #[0,1,2,3] [6,5,32,32]
  #print(dim_list)

  #pdb.set_trace()


  #元のメソッドにデコレーターの@tf.functionがついているためTrueにならない
  ave, sig = tf.nn.moments(target,dim_list)


  if get_sig:
    return sig
  #sig = 0.5
  #sig = tf.math.add(sig,0.5)

  #sig = tf.math.square(sig,name='nijou')

  pdb.set_trace()
  #noise = np.random.normal(0,sig,target.shape)
  noise = tf.random.normal(target.shape,stddev=sig,dtype=target.dtype)

  #print(noise)
  t_added = tf.math.add(target, noise)
  #pdb.set_trace()
  #t_added = noise + target
  #t_added  = tf.convert_to_tensor(t_added, dtype=tf.float16)

  # npに変換して足していたがtensorを保持したまま足したい
  #tf.config.experimental_run_functions_eagerly(False)
  return t_added


'''
arr =  tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float16)
arr_1 = add_noise(arr)


print(arr)
print(arr_1)'''
