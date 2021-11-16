"""   YPL & JLL, 2021.11.16
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
import numpy as np
import matplotlib.pyplot as plt

def datagen(cf5X_data, pf5P_data, rf5L_data, batch_size):
    X_batch = np.zeros((batch_size, 12, 128, 256), dtype='float32')   # for YUV imgs
    Y_batch = np.zeros((batch_size, 112), dtype='float32')
    batchIndx = 0
    Nplot = 0
    dataN = len(cf5X_data)

    while True:
        count = 0
        while count < batch_size:
            ri = np.random.randint(0, dataN-2, 1)[-1]   # ri cannot be the last dataN-1
            print("#---  count, dataN, ri =", count, dataN, ri)
            for i in range(dataN-1):
              if ri < dataN-1:
                vsX1 = cf5X_data[ri]
                vsX2 = cf5X_data[ri+1]
                #---  vsX2.shape = (6, 128, 256)
                X_batch[count] = np.vstack((vsX1, vsX2))
                #---  X_batch[count].shape = (12, 128, 256)

                pf5P1 = pf5P_data[ri]
                pf5P2 = pf5P_data[ri+1]
                rf5L1 = rf5L_data[ri]
                rf5L2 = rf5L_data[ri+1]
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
            #print('#---datagenB3  Yb =', Yb)
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
