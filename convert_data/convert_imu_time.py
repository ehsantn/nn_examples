import rosbag
import numpy as np
import glob


dset_dir = '/export/datasets/adg/'
dset_name = 'BXP5401-imu_2017-*.bag'

files = glob.glob(dset_dir + dset_name)
num_vals = 0

for fname in files:
    bag = rosbag.Bag(fname)
    num_vals += bag.get_message_count('/imu/imu_ahrs')
    bag.close()

T = np.empty(num_vals, np.int64)

i = 0
for fname in files:
    bag = rosbag.Bag(fname)
    for topic, msg, t in bag.read_messages('/imu/imu_ahrs'):
        T[i] = t.to_nsec()
        i += 1
    bag.close()

out_dir = "/export/intel/users/etotoni/"

T.tofile(out_dir+"accel_T.dat")
