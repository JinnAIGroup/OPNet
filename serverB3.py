"""   YPL & JLL, 2021.9.24
(YPNet) jinn@Liu:~/YPNet$ python serverB3.py
"""
import os
import zmq
import six
import random
import logging
import argparse
import numpy as np
from numpy.lib.format import header_data_from_array_1_0
from datagenB3 import datagen

all_dirs = os.listdir('/home/jinn/dataB')
all_videos = ['/home/jinn/dataB/'+i+'/video.hevc' for i in all_dirs]
#print('#---  all_videos =', all_videos)
'''--- all_videos = [
'/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/video.hevc',
'/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/video.hevc']
'''

if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

logger = logging.getLogger(__name__)

random.seed(0) # makes the random numbers predictable
random.shuffle(all_videos)

def send_arrays(socket, arrays, stop=False):
  if arrays:
    # The buffer protocol only works on contiguous arrays
    arrays = [np.ascontiguousarray(array) for array in arrays]
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
    array = np.frombuffer(buf, dtype=np.dtype(header['descr']))
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

def start_server(data_obj, port=5557, hwm=20):
  logging.basicConfig(level='INFO')
  context = zmq.Context()
  socket = context.socket(zmq.PUSH)
  socket.set_hwm(hwm)
  socket.bind('tcp://*:{}'.format(port))

  # it = itertools.tee(data_obj)
  it = data_obj
  logger.info(' #--- serverB3 started')
  while True:
    try:
      data_stream = next(it)
      stop = False
      logger.debug("sending {} arrays".format(len(data_stream)))
    except StopIteration:
      it = data_obj
      data_stream = None
      stop = True
      logger.debug("sending StopIteration")

    print('#---serverB3  np.shape(data_stream[0]) =', np.shape(data_stream[0]))
    print('#---serverB3  np.shape(data_stream[1]) =', np.shape(data_stream[1]))
    #print('#---serverB3  data_stream =', data_stream)
    send_arrays(socket, data_stream, stop=stop)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Data Server')
  parser.add_argument('--batch', dest='batch', type=int, default=25, help='Batch size')
  parser.add_argument('--time', dest='time', type=int, default=2, help='Number of frames per sample')
  parser.add_argument('--port', dest='port', type=int, default=5557, help='Port of the ZMQ server')
  parser.add_argument('--buffer', dest='buffer', type=int, default=20, help='High-water mark. Increasing this increses buffer and memory usage.')
  parser.add_argument('--validation', dest='validation', action='store_true', default=False, help='Serve validation dataset instead.')
  args, more = parser.parse_known_args()

  train_len  = int(0.5*len(all_videos))
  valid_len  = int(0.5*len(all_videos))
  train_videos = all_videos[: train_len]
  valid_videos = all_videos[train_len: train_len + valid_len]
  print('#---serverB3  len(all_videos) =', len(all_videos))
  print('#---serverB3  len(train_videos) =', len(train_videos))
  print('#---serverB3  len(valid_videos) =', len(valid_videos))

  if args.validation:
    video_files = valid_videos
  else:
    video_files = train_videos

  print('#---serverB3  video_files =', video_files)
  #---  video_files = ['/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/video.hevc']
  data_obj = datagen(video_files, max_time_len=args.time, batch_size=args.batch)
  print('#---serverB3  np.shape(data_obj) =', np.shape(data_obj))
  start_server(data_obj, port=args.port, hwm=args.buffer)
