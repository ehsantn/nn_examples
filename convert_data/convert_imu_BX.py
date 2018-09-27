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


X = np.empty(num_vals)
Y = np.empty(num_vals)
Z = np.empty(num_vals)
aX = np.empty(num_vals)
aY = np.empty(num_vals)
aZ = np.empty(num_vals)

i = 0
for fname in files:
    bag = rosbag.Bag(fname)
    for topic, msg, t in bag.read_messages('/imu/imu_ahrs'):
        X[i] = msg.linear_acceleration.x
        Y[i] = msg.linear_acceleration.y
        Z[i] = msg.linear_acceleration.z
        aX[i] = msg.angular_velocity.x
        aY[i] = msg.angular_velocity.y
        aZ[i] = msg.angular_velocity.z
        i += 1
    bag.close()

out_dir = "/export/intel/users/etotoni/"

X.tofile(out_dir+"accel_X.dat")
Y.tofile(out_dir+"accel_Y.dat")
Z.tofile(out_dir+"accel_Z.dat")

aX.tofile(out_dir+"ang_X.dat")
aY.tofile(out_dir+"ang_Y.dat")
aZ.tofile(out_dir+"ang_Z.dat")
