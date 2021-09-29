"""   YPL & JLL, 2021.9.24
"""
import os
import cv2
import time
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

def concatenate(video_files):
  camera_files  = [f.replace('video.hevc', 'camera320.h5') for f in video_files]
  path_files  = [f.replace('video.hevc', 'pathdata.h5') for f in video_files]
  radar_files = [f.replace('video.hevc', 'radardata.h5') for f in video_files]
  print('#---datagenB3  path_files =', path_files)
  #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']

  lastidx = 0
  all_frames = []
  path = []
  lead = []
  for vfile, cfile, pfile, rfile in zip(video_files, camera_files, path_files, radar_files):
  # Traversing Lists in Parallel, https://realpython.com/python-zip-function/
    if os.path.isfile(vfile):
      with h5py.File(pfile, "r") as pf5:
        t0 = time.time()
        cf5 = h5py.File(cfile, 'r')
        rf5 = h5py.File(rfile, 'r')

        Path = pf5['Path'][:]
        path.append(Path)
        t1 = time.time()

        t2 = time.time()
        leadOne = rf5['LeadOne'][:]
        lead.append(leadOne)

        x = cf5['X']
        all_frames.append([lastidx, lastidx+cf5['X'].shape[0], x])
        #print("x {} ".format(x.shape[0]))
        lastidx += cf5['X'].shape[0]

  path = np.concatenate(path, axis=0)
  lead = np.concatenate(lead, axis=0)
  #---  path.shape = (1150, 51)
  #---  lead.shape = (1150, 5)
  #---  len(lead) = 1150

  out = np.concatenate((path, lead), axis=-1)
  out[:, 50:52]/=10
  #print('  #---  oout[:, 50:52] =', out[:, 50:52])
  print ("#---datagenB3  total training %d frames" % (lastidx))
  print ("#---datagenB3  np.shape(all_frames) =", np.shape(all_frames))
  #print ("  #---  all_frames.shape =", all_frames.shape)

  #---  np.shape(all_frames) = (1, 3)
  #---  all_frames = [[0, 1150, <HDF5 dataset "X": shape (1150, 160, 320, 3), type "<f4">]]
  #---  lastidx = 1150
  #---  out.shape = (1150, 56)
  return all_frames, lastidx, out

def datagen(video_files, max_time_len, batch_size):
  all_frames, lastidx, out = concatenate(video_files)
  X_batch = np.zeros((batch_size, 12, 128, 256), dtype='uint8')
  lead_path_batch = np.zeros((batch_size, max_time_len, 51+5), dtype='float32')
  #---  np.shape(X_batch) = (25, 2, 160, 320, 3)
  #---  np.shape(lead_path_batch) = (25, 2, 56)
  #print ("#---  np.shape(lead_path_batch) =", np.shape(lead_path_batch))

  while True:
    t = time.time()
    count = 0
    time_len = 2
    frames_b_t = X_batch[:, :time_len]
    lead_path_batch_t = lead_path_batch[:, :time_len]
    while count < batch_size:
      #print(ri)
      ri = np.random.randint(0, lastidx, 1)[-1]
      for fl, el, x in all_frames:
        if fl <= ri and ri < el:
          if el-ri < time_len:
            rj=np.random.randint(time_len-(el-ri),ri-fl,1)[-1]
            #print(rj, outindex, ri[count])
            frames_b_t[count] = x[ri-rj-fl: ri-rj-fl+time_len] #if time_len>1
            lead_path_batch_t[count] = out[ri-rj: ri-rj+time_len]
            #print(fl, ri-rj-fl)
          else:
            #print(fl, ri-fl)
            frames_b_t[count] = x[ri-fl: ri-fl+time_len]
            lead_path_batch_t[count] = out[ri: ri+time_len]
          break

      #print(lead_path_batch[count])
      count += 1
    print('#---datagenB3  count =', count)

    #print(lead_path_batch[2][:, -1])
    t2 = time.time()
    a = np.hstack(X_batch[0])
    #---  np.shape(X_batch[0]) = (2, 160, 320, 3)
    print ("#---datagenB3  np.shape(X_batch[0]) =", np.shape(X_batch[0]))
    plt.imshow(a)
    plt.show()
    print('#---datagenB3  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))

    print ("#---datagenB3  np.shape(frames_b_t) =", np.shape(frames_b_t))
    print ("#---datagenB3  np.shape(lead_path_batch_t) =", np.shape(lead_path_batch_t))
    #---  np.shape(frames_b_t), np.shape(lead_path_batch_t) = (25, 2, 160, 320, 3) (25, 2, 56)
    yield(frames_b_t, lead_path_batch_t)
