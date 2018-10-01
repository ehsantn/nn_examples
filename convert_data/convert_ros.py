import rosbag
import glob

dset_dir = '/export/datasets/adg/'
dset_name = 'BXP5401-front-camera_*.bag'

# dset_dir = './'
# fname = 'image_test.bag'

out_fname = "/export/intel/users/etotoni/BXP5401-front-camera_2017_sorted.dat"
f = open(out_fname, 'w')


files = glob.glob(dset_dir + dset_name)
files.sort()
num_img = 0
height = -1
width = -1

for fname in files:
    bag = rosbag.Bag(fname)
    for topic, msg, t in bag.read_messages():
        if msg.height != height or msg.width != width:
            height = msg.height
            width = msg.width
            print(height, width)
        #print(topic, type(msg), t)
        num_img += 1
        f.write(msg.data)

f.close()
print(num_img)
