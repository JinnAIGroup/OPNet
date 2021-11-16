"""   YPL & JLL, 2021.11.16
from /home/jinn/YPN/YPNetB/serverB3.py

Run: on 2 terminals
  (YPN) jinn@Liu:~/YPN/OPNet$ python serverB3.py --port 5557
  (YPN) jinn@Liu:~/YPN/OPNet$ python serverB3.py --port 5558 --validation
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
import zmq
import six
import numpy
import random
import logging
import argparse
from datagenB3 import datagen
from numpy.lib.format import header_data_from_array_1_0

BATCH_SIZE = 16

all_dirs = os.listdir('/home/jinn/dataB')
all_yuvs = ['/home/jinn/dataB/'+i+'/yuv.h5' for i in all_dirs]
  #print('#---  all_yuvs =', all_yuvs)
random.seed(0) # makes the random numbers predictable
random.shuffle(all_yuvs)

path_files  = [f.replace('yuv', 'pathdata') for f in all_yuvs]
radar_files = [f.replace('yuv', 'radardata') for f in all_yuvs]
  #---  path_files  = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/pathdata.h5']
  #---  radar_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/radardata.h5']

for cfile, pfile, rfile in zip(all_yuvs, path_files, radar_files):
    if os.path.isfile(cfile) and os.path.isfile(pfile) and os.path.isfile(rfile):
          #print("#---  cfile =", cfile)
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
            imgsN = len(cf5X)
              #print("#---datagenB3  imgsN =", imgsN)

            train_len  = int(0.9*imgsN)
            valid_len  = int(0.1*imgsN)

            train_cf5X = cf5X[: train_len]
            valid_cf5X = cf5X[train_len: train_len + valid_len]
            train_pf5P = pf5P[: train_len]
            valid_pf5P = pf5P[train_len: train_len + valid_len]
            train_rf5L = rf5L[: train_len]
            valid_rf5L = rf5L[train_len: train_len + valid_len]
    else:
        print('#---serverB3  Error: cfile, pfile, or rfile does not exist')


if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

logger = logging.getLogger(__name__)

def send_arrays(socket, arrays, stop=False):
  if arrays:
      # The buffer protocol only works on contiguous arrays
    arrays = [numpy.ascontiguousarray(array) for array in arrays]
  if stop:
    headers = {'stop': True}
    socket.send_json(headers)
  else:
    headers = [header_data_from_array_1_0(array) for array in arrays]
    socket.send_json(headers, zmq.SNDMORE)
    for array in arrays[:-1]:
      socket.send(array, zmq.SNDMORE)
    socket.send(arrays[-1])

def recv_arrays(socket):
  headers = socket.recv_json()
  if 'stop' in headers:
    raise StopIteration

  arrays = []
  for header in headers:
    data = socket.recv()
    buf = buffer_(data)
    array = numpy.frombuffer(buf, dtype=numpy.dtype(header['descr']))
    array.shape = header['shape']
    if header['fortran_order']:
      array.shape = header['shape'][::-1]
      array = array.transpose()
    arrays.append(array)

  return arrays

def client_generator(port=5557, host="localhost", hwm=20):
  context = zmq.Context()
  socket = context.socket(zmq.PULL)
  socket.set_hwm(hwm)
  socket.connect("tcp://{}:{}".format(host, port))
  logger.info('client started')
  while True:
    data = recv_arrays(socket)
    yield tuple(data)

def start_server(data_stream, port=5557, hwm=20):
  logging.basicConfig(level='INFO')
  context = zmq.Context()
  socket = context.socket(zmq.PUSH)
  socket.set_hwm(hwm)
  socket.bind('tcp://*:{}'.format(port))

    # it = itertools.tee(data_stream)
  it = data_stream
  logger.info('server started')
  while True:
    try:
      data = next(it)
      stop = False
      logger.debug("sending {} arrays".format(len(data)))
    except StopIteration:
      it = data_stream
      data = None
      stop = True
      logger.debug("sending StopIteration")

    send_arrays(socket, data, stop=stop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Server')
    parser.add_argument('--port', dest='port', type=int, default=5557, help='Port of the ZMQ server')
    parser.add_argument('--buffer', dest='buffer', type=int, default=20, help='High-water mark. Increasing this increses buffer and memory usage.')
    parser.add_argument('--validation', dest='validation', action='store_true', default=False, help='Serve validation dataset instead.')
    args, more = parser.parse_known_args()

    if args.validation:
      cf5X_data = valid_cf5X
      pf5P_data = valid_pf5P
      rf5L_data = valid_rf5L
    else:
      cf5X_data = train_cf5X
      pf5P_data = train_pf5P
      rf5L_data = train_rf5L

    data_s = datagen(cf5X_data, pf5P_data, rf5L_data, BATCH_SIZE)
    start_server(data_s, port=args.port, hwm=args.buffer)
