"""   YPL & JLL, 2021.9.15, 10.9
Make yuv.h5 by /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py

Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5
Output:
  X_batch.shape = (none, 12, 128, 256)
  Y_batch.shape = (none, 2, 56)
"""
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt

def concatenate(camera_files):
  path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
  radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
    #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
    #---  radar_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']
  lastidx = 0
  all_frames = []
  path = []
  lead = []

  for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
    # zip: Traversing Lists in Parallel, https://realpython.com/python-zip-function/
    if os.path.isfile(rfile):
      with h5py.File(pfile, "r") as pf5:
        cf5 = h5py.File(cfile, 'r')
        rf5 = h5py.File(rfile, 'r')
        print("#---datagenB3  cf5['X'].shape       =", cf5['X'].shape)
        print("#---datagenB3  pf5['Path'].shape    =", pf5['Path'].shape)
        print("#---datagenB3  rf5['LeadOne'].shape =", rf5['LeadOne'].shape)

        Path = pf5['Path'][:]
        path.append(Path)

        leadOne = rf5['LeadOne'][:]
        lead.append(leadOne)

          #---  cf5['X'].shape       = (1200, 6, 128, 256)
          #Xcf5 = cf5['X']
          #Xcf5 = cf5['X'][0:1150]   # NG, this gives different print('#---  all_frames =', all_frames)
          # Conclusion: make cf5['X'].shape = (1150, 6, 128, 256) by
          #   /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py
        Xcf5 = cf5['X']
          #print("#---datagenB3  Xcf5.shape =", Xcf5.shape)
        all_frames.append([lastidx, lastidx+cf5['X'].shape[0], Xcf5])
        lastidx += Xcf5.shape[0]

  path = np.concatenate(path, axis=0)
  lead = np.concatenate(lead, axis=0)
    #---  path.shape = (1150, 51)
    #---  lead.shape = (1150, 5)
    #---  len(lead) = 1150

  PnL = np.concatenate((path, lead), axis=-1)
  PnL[:, 50:52]/=10
    #print('  #---  PnL[:, 50:52] =', PnL[:, 50:52])
  print("#---datagenB3  total training %d frames" % (lastidx))

  print('#---datagenB3  all_frames =', all_frames)
    #print('#---datagenB3  np.shape(all_frames) =', np.shape(all_frames))
    #print('#---datagenB3  lastidx =', lastidx)
  print('#---datagenB3  PnL.shape =', PnL.shape)
    #---  np.shape(all_frames) = (1, 3)
    #---  all_frames = [[0, 1150, <HDF5 dataset "X": shape (1150, 6, 128, 256), type "<f4">]]
    #---  lastidx = 1150
    #---  PnL.shape = (1150, 56)
  return all_frames, lastidx, PnL

def datagen(camera_files, max_time_len, batch_size=10):  # I did not use max_time_len
  all_frames, lastidx, PnL = concatenate(camera_files)
  X_batch = np.zeros((batch_size, 12, 128, 256), dtype='uint8')
  Y_batch = np.zeros((batch_size, 2, 51+5), dtype='float32')

  Nplot = 0
  while True:
    t = time.time()
    count = 0

    while count < batch_size:
      ri = np.random.randint(0, lastidx-1, 1)[-1]  # lastidx-1 = 1149

      for fl, el, Xcf5 in all_frames:
        #---  Xcf5.shape = (1150, 6, 128, 256)

        if fl <= ri and ri < el:
          print('#---datagenB3  fl, el, ri =', fl, el, ri)
          print('#---datagenB3  ri-fl =', ri-fl)
            #---  fl, el, ri = 0 1150 345
            #---  ri-fl = 345
          vsX1 = Xcf5[ri-fl]
          vsX2 = Xcf5[ri-fl+1]
            #---  vsX2.shape = (6, 128, 256)
          X_batch[count] = np.vstack((vsX1, vsX2))
            #---  X_batch[count].shape = (12, 128, 256)

          vsPnL1 = PnL[ri-fl]
          vsPnL2 = PnL[ri-fl+1]
            #---  vsO2.shape = (56,)
          Y_batch[count] = np.vstack((vsPnL1, vsPnL2))
            #---  Y_batch[count].shape = (2, 56)
          break

      print('#---  count =', count)
      print('#---  not repeated? ri =', ri)
      count += 1

    if Nplot == 0:
      Yb = Y_batch[0][0][:]
      print('#---datagenB3  Yb.shape =', Yb.shape)
      print('#---datagenB3  Yb =', Yb)
      print('#---datagenB3  Y_batch[0][0][50:52] =', Y_batch[0][0][50:52])
      plt.plot(Yb)
      plt.show()
      Nplot += 1


      # y_t54 = y_true[:, 50:55]  # true (lead car) lcar's d, y, v, a
    Y_batch[:, :, 50:52]/=10   # normalize 50 (valid_len), 51 (lcar's d)
    if Nplot == 1:
      Yb = Y_batch[0][0][:]
      print('#---datagenB3  Yb.shape =', Yb.shape)
      print('#---datagenB3  Yb =', Yb)
      print('#---datagenB3  Y_batch[0][0][50:52] =', Y_batch[0][0][50:52])
      plt.plot(Yb)
      plt.show()
      Nplot += 1

    t2 = time.time()
    print('#---datagenB3  datagen time =', "%5.2f ms" % ((t2-t)*1000.0))
    print('#---datagenB3  X_batch.shape =', X_batch.shape)
    print('#---datagenB3  Y_batch.shape =', Y_batch.shape)
      #---  X_batch.shape = (25, 12, 128, 256)
      #---  Y_batch.shape = (25, 2, 56)
    yield(X_batch, Y_batch)
