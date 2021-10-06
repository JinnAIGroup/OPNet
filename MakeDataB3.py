'''  JLL, 2021.10.5-6
From /home/jinn/openpilot/tools/lib/hevctoh5old.py
     /home/jinn/openpilot/tools/lib/bz2toh5.py

(OP082) jinn@Liu:~/openpilot/tools/lib$ python MakeDataB3.py
Input:
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/raw_log.bz2
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/global_pose/frame_times
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/global_pose/frame_positions
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/global_pose/frame_orientations
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/global_pose/frame_velocities
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/raw_log.bz2
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/global_pose/frame_times
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/global_pose/frame_positions
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/global_pose/frame_orientations
/home/jinn/dataB/UHD--2018-08-02--08-34-47--33/global_pose/frame_velocities
Output:
#---  np.shape(NradarData) = (2, 1150, 5)
#---  np.shape(NpathData)  = (2, 1150, 51)
'''
import os
import h5py
import numpy as np
from tools.lib.logreader import LogReader
import orientation as orient  # /home/jinn/openpilot/tools/lib/orientation.py
import coordinates as coord   # /home/jinn/openpilot/tools/lib/coordinates.py

def MakeData(all_dirs):
  NradarData = []
  NpathData = []
  for dir in all_dirs:
    lr = LogReader(dir + 'raw_log.bz2')
    print('#---  raw_log.bz2 =', dir + 'raw_log.bz2')
    logs = list(lr)  #---  len(logs) = 69061  len(new_list)  = 37

    CameraS_t = np.array([l.logMonoTime*10**-9 for l in logs if l.which()=='roadCameraState'])
    RadarS_t  = np.array([l.logMonoTime*10**-9 for l in logs if l.which()=='radarState'])
    RadarS = [l.radarState.leadOne for l in logs if l.which()=='radarState']
    print("#---  len(CameraS_t) =", len(CameraS_t))
    print("#---  len(RadarS_t) =", len(RadarS_t))
    print("#---  len(RadarS) =", len(RadarS))
    frame_times = np.load(dir + 'global_pose/frame_times')
    frame_positions = np.load(dir + 'global_pose/frame_positions')
    frame_orientations = np.load(dir + 'global_pose/frame_orientations')
    frame_velocities = np.load(dir + 'global_pose/frame_velocities')
    velocities = np.linalg.norm(frame_velocities,axis=1)
    #print(velocities)

    Nframe_t = len(frame_times)
    print("#---  Nframe_t =", Nframe_t)
    Nframes = max(len(CameraS_t), len(RadarS_t)) - 50
    print("#---  Nframes =", Nframes)
    mRadarS = []

    if Nframes > 0:
      mRadarS_t = []
      for t in CameraS_t:
        minindex = np.argmin(np.abs(RadarS_t-t))
        mRadarS_t.append(RadarS_t[minindex])
        mRadarS.append(RadarS[minindex])
      er = np.abs(mRadarS_t-CameraS_t)
      if (er < 0.05).all():
        RadarData = np.zeros((Nframes, 5), dtype='uint8')
        PathData = np.zeros((Nframes, 51), dtype='uint8')
        for i in range(Nframes):
          d = mRadarS[i].dRel
          y = mRadarS[i].yRel
          v = mRadarS[i].vRel
          a = mRadarS[i].aRel
          if a==0 and y==0 and v==0 and d==0:
            prob = 0
          else:
            prob = 1
          RadarData[i] = [d, y, v, a, prob]

          ecef_from_local = orient.rot_from_quat(frame_orientations[i])
          local_from_ecef = ecef_from_local.T
          frame_positions_local = np.einsum('ij,kj->ki', local_from_ecef, frame_positions[i:] - frame_positions[i])
          #print(len(frame_positions_local))
          x_f = frame_positions_local[:50,0]
          y_f = frame_positions_local[:50,1]
          linear_model = np.polyfit(x_f,y_f,3)
          linear_model_fn = np.poly1d(linear_model)
          vaild_len = x_f[-1]
          x_e = np.array([1+i for i in range(50)])
          y_e = linear_model_fn(x_e)
          PathData[i] = np.hstack([y_e, vaild_len])
      else:
        print("??? Error  (er < 0.05).all() =", (er < 0.05).all())
    else:
      print("??? Error  Nframes =", Nframes)
    NradarData.append(RadarData)
    NpathData.append(PathData)
    #---  RadarData.shape = (1150, 5)
    print('#---  PathData.shape =', PathData.shape)

    print("#---  len(mRadarS_t) =", len(mRadarS_t))
    print("#---  len(mRadarS) =", len(mRadarS))
    #fh5 = h5py.File(radar_file, 'r') # fh5 = radar_file = radardata.h5
    #--- list(fh5.keys()) = ['LeadOne']
    #dataset = fh5['LeadOne']
    #--- dataset.shape = (10, 5)  # 10 = 1201 - 1191 (10 frames)

  return NradarData, NpathData

if __name__ == "__main__":
  all_dirs = os.listdir('/home/jinn/dataB')
  all_dirs = ['/home/jinn/dataB/'+i+'/' for i in all_dirs]
  #print('#---  all_dirs =', all_dirs)

  NradarData, NpathData = MakeData(all_dirs)
  print("#---  np.shape(NradarData) =", np.shape(NradarData))
  print("#---  np.shape(NpathData) =", np.shape(NpathData))
  #---  np.shape(NradarData) = (2, 1150, 5)
  #---  np.shape(NpathData)  = (2, 1150, 51)

  #for vf, lf in zip(all_videos, all_logs):
  #  reader(vf, lf)

'''
/home/jinn/dataB/UHD--2018-08-02--08-34-47--32/raw_log.bz2
    new_list = [l.which() for l in logs]
    new_list = list(set(new_list)) # get unique items (to set and back to list)
    print("#--- len(new_list) =", len(new_list))
    print("#--- new_list =", new_list)
#---  len(new_list) = 28
#---  new_list =
['androidLog', 'carControl', 'carState', 'can', 'clocks', 'controlsState',
'deviceState', 'driverState', 'gpsLocation', 'gpsLocationExternal', 'gpsNMEA',
'liveCalibration', 'liveLongitudinalMpc', 'liveMpc', 'liveTracks', 'logMessage',
'longitudinalPlan', 'model', 'pandaState', 'procLog', 'qcomGnssDEPRECATD',
'radarState', 'roadCameraState', 'roadEncodeIdx',
'sendcan', 'sensorEvents', 'ubloxGnss', 'ubloxRaw']

/home/jinn/dataA/8bfda98c9c9e4291%7C2020-05-11--03-00-57--61/rlog.bz2
#---  len(new_list)  = 37
['androidLog', 'cameraOdometry', 'can', 'carControl', 'carEvents', 'carParams', 'carState',
'clocks', 'controlsState', 'deviceState', 'driverCameraState', 'driverMonitoringState', 'driverState',
'gpsLocation', 'gpsNMEA', 'gpsLocationExternal', 'initData',
'lateralPlan', 'liveCalibration', 'liveLocationKalman', 'liveParameters', 'liveTracks',
'logMessage', 'longitudinalPlan', 'model', 'pandaState', 'procLog', 'qcomGnssDEPRECATD',
'radarState', 'roadCameraState', 'roadEncodeIdx',
'sendcan', 'sensorEvents', 'sentinel', 'thumbnail', 'ubloxGnss', 'ubloxRaw']
'''
