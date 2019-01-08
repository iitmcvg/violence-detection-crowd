from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
# import ConfigParser
import collections
import time
import csv
import os
import cv2
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
from Anom_detect.Test_Anomaly_Detector_public import model_setup,test_video
# import theano.sandbox
import shutil

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# import C3D_tensorflow.input_data
import C3D_tensorflow.c3d_model
import C3D_tensorflow.input_pipe
from C3D_tensorflow.predict_c3d_ucf101 import placeholder_inputs
from C3D_tensorflow.predict_c3d_ucf101 import integ_test_c3d
import numpy as np

flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS

cap = cv2.VideoCapture('./test.mp4')
gpu_num = 1

model_name = './C3D_tensorflow/c3d_ucf101_finetune_whole_iter_20000_TF.model'

images_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
sess,features = integ_test_c3d(FLAGS,images_placeholder)
saver = tf.train.Saver()
saver.restore(sess,model_name)

model_pred = model_setup()
mean = np.load('./C3D_tensorflow/crop_mean.npy')

cv2.namedWindow('Analysed',cv2.WINDOW_AUTOSIZE)
i = 0
while True:
    feature_arr = np.zeros([1,4096])
    # for i in range(32):
    frames_arr = np.zeros([1,16,112,112,3])
    for k in range(16):
        ret,frame1 = cap.read()
        frame = cv2.resize(frame1,(112,112))
        frames_arr[0,k,:,:,:] = frame.astype('float')
    frames_arr = frames_arr-mean
    feats = sess.run(features ,feed_dict={images_placeholder: frames_arr})
    feature_arr[0,:] = feats
    
    pred = np.int(test_video(feature_arr,model_pred)[0,0])
    if pred == 1:
        text = "Violence"
        format = np.zeros_like(frame1)
        format[:,:,2] = 255
        colour = (0,0,255)
    else:
        text = "No Violence"
        format = np.zeros_like(frame1)
        format[:,:,1] = 255
        colour = (0,255,0)
    flag = 0.3
    frame1 = flag*format.astype('uint8') + (1-flag)*frame1
    # frame = cv2.resize(frame1,None,fx=4,fy=4)
    # print(frame1.shape)
    cv2.putText(frame1,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,2,colour,3)
    cv2.imshow('Analysed',frame1.astype('uint8')) 
    cv2.waitKey(10)
    cv2.imwrite('./output_images/'+str(i)+'.jpg',frame1.astype('uint8'))   
    i = i+1