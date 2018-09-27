import rosbag
import glob
import numpy as np

dset_dir = '/export/datasets/adg/'
dset_name = 'BXP5401-front-camera_*.bag'

# dset_dir = './'
# fname = 'image_test.bag'

out_fname = '/export/datasets/imgs_time.dat'
#f = open(out_fname, 'wb')


files = glob.glob(dset_dir + dset_name)
num_img = 0
data = []

for fname in files:
    bag = rosbag.Bag(fname)
    for topic, msg, t in bag.read_messages():
        #print(topic, type(msg), t)
        num_img += 1
        #f.write(t.to_nsec())
        data.append(t.to_nsec())

#f.close()
print(num_img)
A = np.array(data, np.int64)
A.tofile(out_fname)