"""   YPL & JLL, 2021.11.17
from /home/jinn/YPN/OPNet/datagenA4.py and datagenB3a.py
solve ??? in datagenA4a.py
make yuv.h5 by /home/jinn/openpilot/tools/lib/hevc2yuvh5B3.py

Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/yuv.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5
  /home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5
Output:
  X_batch.shape = (none, 2x6, 128, 256)  (num_channels = 6, 2 yuv images)
  Y_batch.shape = (none, 2x56)  (56 = 51 pathdata + 5 radardata)
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def datagen(camera_files, batch_size):
    X_batch = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # for YUV imgs
    Y_batch = np.zeros((batch_size, 112), dtype='float32')
    path_files  = [f.replace('yuv', 'pathdata') for f in camera_files]
    radar_files = [f.replace('yuv', 'radardata') for f in camera_files]
      #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
      #---  radar_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']
    for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
        if os.path.isfile(cfile) and os.path.isfile(pfile) and os.path.isfile(rfile):
            print('#---serverB3  OK: cfile, pfile, rfile exist')
        else:
            print('#---serverB3  Error: cfile, pfile, or rfile does not exist')
    batchIndx = 0
    Nplot = 0

    while True:
        for cfile, pfile, rfile in zip(camera_files, path_files, radar_files):
            with h5py.File(cfile, "r") as cf5:
                pf5 = h5py.File(pfile, 'r')
                rf5 = h5py.File(rfile, 'r')
                  #---  cf5['X'].shape       = (1150, 6, 128, 256)
                  #---  pf5['Path'].shape    = (1150, 51)
                  #---  rf5['LeadOne'].shape = (1150, 5)
                cf5X = cf5['X']
                pf5P = pf5['Path']
                rf5L = rf5['LeadOne']
                  #---  cf5X.shape = (1150, 6, 128, 256)

                dataN = len(cf5X)

                count = 0
                while count < batch_size:
                    ri = np.random.randint(0, dataN-2, 1)[-1]   # ri cannot be the last dataN-1
                    print("#---  count, dataN, ri =", count, dataN, ri)
                    for i in range(dataN-1):
                      if ri < dataN-1:
                        vsX1 = cf5X[ri]
                        vsX2 = cf5X[ri+1]
                          #---  vsX2.shape = (6, 128, 256)
                        X_batch[count] = np.vstack((vsX1, vsX2))
                          #---  X_batch[count].shape = (12, 128, 256)

                        pf5P1 = pf5P[ri]
                        pf5P2 = pf5P[ri+1]
                        rf5L1 = rf5L[ri]
                        rf5L2 = rf5L[ri+1]
                          #---  pf5P2.shape = (51,)
                          #---  rf5L2.shape = (5,)
                        Y_batch[count] = np.hstack((pf5P1, rf5L1, pf5P2, rf5L2))
                          #---  Y_batch[count].shape = (112,)
                        break
                    count += 1

                batchIndx += 1
                print('#---  batchIndx =', batchIndx)

                if Nplot == 0:
                    Yb = Y_batch[0][:]
                    print('#---datagenB3  Yb.shape =', Yb.shape)
                    print('#---datagenB3  Y_batch[0][50:52] =', Y_batch[0][50:52])
                    print('#---datagenB3  Y_batch[0][106:108] =', Y_batch[0][106:108])
                    plt.plot(Yb)
                    plt.show()
                    Nplot += 1

                Y_batch[:, 50:52]/=100   # normalize 50 (valid_len), 51 (lcar's d)
                Y_batch[:, 106:108]/=100

                if Nplot == 1:
                    Yb = Y_batch[0][:]
                    print('#---datagenB3  Y_batch[0][50:52] =', Y_batch[0][50:52])
                    print('#---datagenB3  Y_batch[0][106:108] =', Y_batch[0][106:108])
                    plt.plot(Yb)
                    plt.show()
                    Nplot += 1

                  #print('#---datagenB3  X_batch.shape =', X_batch.shape)
                  #print('#---datagenB3  Y_batch.shape =', Y_batch.shape)
                  #---  X_batch.shape = (25, 12, 128, 256)
                  #---  Y_batch.shape = (25, 112)
                yield(X_batch, Y_batch)
