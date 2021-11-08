"""   YPL & JLL, 2021.11.2, 11.8
(YPN) jinn@Liu:~/YPN/OPNet$ python serverA4.py --port 5557
(YPN) jinn@Liu:~/YPN/OPNet$ python serverA4.py --port 5558 --validation
use 2 terminals

Input:
  bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
  sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
  /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5  (X for debugging with 10 imgs)
  /home/jinn/dataAll/comma10k/Xmasks/*.png
Output:
  X_batch.shape = (none, 2x6, 128, 256) (num_channels = 6, 2 yuv images)
  Y_batch.shape = (none, 256, 512, 2x6) (num_classes = 6)
"""
import os
import zmq
import six
import numpy
import random
import logging
import argparse
from datagenA4 import datagen
from numpy.lib.format import header_data_from_array_1_0

DATA_DIR_Imgs = '/home/jinn/dataAll/comma10k/Ximgs_yuv/'  # Ximgs with 10 images only for debugging
DATA_DIR_Msks = '/home/jinn/dataAll/comma10k/Xmasks/'
BATCH_SIZE = 2

IMAGE_H = 128
IMAGE_W = 256
MASK_H = 256
MASK_W = 512

class_values = [41,  76,  90, 124, 161, 0] # 0 added for padding
# https://github.com/YassineYousfi/comma10k-baseline/blob/ca1c0d1f47e5c4cb14f7ab29130d8f20dec5fc87/LitModel.py

all_img_dirs = os.listdir(DATA_DIR_Imgs)
all_msk_dirs = os.listdir(DATA_DIR_Msks)
all_images = [DATA_DIR_Imgs+i for i in all_img_dirs]
all_masks = [DATA_DIR_Msks+i for i in all_msk_dirs]
all_images = sorted(all_images)
all_masks = sorted(all_masks)
  #print('#---1  all_images =', all_images)
  #print('#---2  all_masks =', all_masks)

train_len = int(0.6*len(all_images))
valid_len = int(0.4*len(all_images))
train_images = all_images[: train_len]
valid_images = all_images[train_len: train_len + valid_len]
train_masks = all_masks[: train_len]
valid_masks = all_masks[train_len: train_len + valid_len]
#print('#---serverA4  len(all_images) =', len(all_images))
#print('#---serverA4  len(train_images) =', len(train_images))
#print('#---serverA4  len(valid_images) =', len(valid_images))

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
        images = valid_images
        masks  = valid_masks
    else:
        images = train_images
        masks  = train_masks
    #print('#---serverA4  images =', images)

    data_s = datagen(images, masks, BATCH_SIZE, IMAGE_H, IMAGE_W, MASK_H, MASK_W, class_values)
    #print('#---serverA4  BATCH_SIZE =', BATCH_SIZE)
    start_server(data_s, port=args.port, hwm=args.buffer)

'''
#---1  all_images = [
'/home/jinn/dataAll/comma10k/Ximgs_yuv/0000_0085e9e41513078a_2018-08-19--13-26-08_11_864_yuv.h5',
'/home/jinn/dataAll/comma10k/Ximgs_yuv/0001_a23b0de0bc12dcba_2018-06-24--00-29-19_17_79_yuv.h5',
'/home/jinn/dataAll/comma10k/Ximgs_yuv/0002_e8e95b54ed6116a6_2018-09-05--22-04-33_2_608_yuv.h5',
'/home/jinn/dataAll/comma10k/Ximgs_yuv/0003_97a4ec76e41e8853_2018-09-29--22-46-37_5_585_yuv.h5',
'/home/jinn/dataAll/comma10k/Ximgs_yuv/0004_2ac95059f70d76eb_2018-05-12--17-46-52_56_371_yuv.h5', ...]

#---2  all_masks = [
'/home/jinn/dataAll/comma10k/Xmasks/0000_0085e9e41513078a_2018-08-19--13-26-08_11_864.png',
'/home/jinn/dataAll/comma10k/Xmasks/0001_a23b0de0bc12dcba_2018-06-24--00-29-19_17_79.png',
'/home/jinn/dataAll/comma10k/Xmasks/0002_e8e95b54ed6116a6_2018-09-05--22-04-33_2_608.png',
'/home/jinn/dataAll/comma10k/Xmasks/0003_97a4ec76e41e8853_2018-09-29--22-46-37_5_585.png',
'/home/jinn/dataAll/comma10k/Xmasks/0004_2ac95059f70d76eb_2018-05-12--17-46-52_56_371.png', ...]
'''
