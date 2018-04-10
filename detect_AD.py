import numpy as np
import time
import pandas as pd
import hpat
import hpat.distributed_lower
from hpat import prange
hpat.multithread_mode = True


@hpat.jit(locals={'data:return': 'distributed'})
def read_data():
    #fname = "/export/intel/lustre/etotoni/BXP5401-front-camera_2017.dat"
    fname = "/export/intel/users/etotoni/BXP5401-front-camera_2017.dat"
    #fname = "img2.dat"
    blob = np.fromfile(fname, np.uint8)

    # reshape to images
    n_channels = 3
    height = 800
    width = 1280
    n_images = len(blob)//(n_channels*height*width)
    data = blob.reshape(n_images, height, width, n_channels)

    # select every 500 image
    data = data[::500,:,:,:]

    # crop to 600 by 600
    data = data[:,100:-100, 340:-340,:]
    return data

t1 = time.time()
AD_images = read_data()
print("read and preprocessing time:", time.time()-t1)

import tensorflow as tf
if tf.__version__ < '1.4.0':
  raise ImportError('v1.4.* or later needed')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#ROOT_PATH = '/home/etotoni/pse-hpc/python/hpat/nn_examples/mask_rcnn_inception_v2_coco_2018_01_28'
ROOT_PATH = '/homes/etotoni/dev/mask_rcnn_inception_v2_coco_2018_01_28'
PATH_TO_CKPT = ROOT_PATH + '/frozen_inference_graph.pb'

#PATH_TO_LABELS = '/home/etotoni/pse-hpc/python/hpat/nn_examples/models/research/object_detection/data/mscoco_label_map.pbtxt'
PATH_TO_LABELS = '/homes/etotoni/python/models/research/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# load frozen TF model
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# detection
def run_inference(images, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
          for i in range(len(images)):
            detection_boxes = tensor_dict['detection_boxes'][i]
            detection_masks = tensor_dict['detection_masks'][i]
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][i], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, images.shape[1], images.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            tensor_dict['detection_masks'+str(i)] = detection_masks_reframed


      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: images})
      output_dict['num_detections'] = output_dict['num_detections'].astype(np.int64)
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.uint8)

  return output_dict


t1 = time.time()
output_dict = run_inference(AD_images, detection_graph)
print("inference time", time.time()-t1)
hpat.distributed_lower.fix_i_malloc()

t1 = time.time()
for i in range(len(AD_images)):
    vis_util.visualize_boxes_and_labels_on_image_array(
        AD_images[i],
        output_dict['detection_boxes'][i],
        output_dict['detection_classes'][i],
        output_dict['detection_scores'][i],
        category_index,
        instance_masks=output_dict['detection_masks'+str(i)],
        use_normalized_coordinates=True,
        line_thickness=8)

print("visualize time", time.time()-t1)
CAR_CLASS = 3  # TODO: get from data
BIKE_CLASS = 2

@hpat.jit(locals={'imgs:input': 'distributed',
          'num_detections:input': 'distributed',
          'detection_classes:input': 'distributed',
          'detection_scores:input': 'distributed'})
def get_stats(imgs, num_detections, detection_classes, detection_scores):
    n_imgs = len(num_detections)
    car_mask = np.empty(n_imgs, np.bool_)

    top_detect_scores = detection_scores[:,0]
    num_cars = np.empty(n_imgs, np.int32)
    num_bikes = np.empty(n_imgs, np.int32)
    for i in prange(n_imgs):
        car_mask[i] = False
        num_cars[i] = 0
        num_bikes[i] = 0
        classes = detection_classes[i,:]
        for j in range(len(classes)):
            if classes[j] == CAR_CLASS:
                num_cars[i] += 1
                car_mask[i] = True
            if classes[j] == BIKE_CLASS:
                num_bikes[i] += 1

    df = pd.DataFrame({'top_detect_scores': top_detect_scores,
                       'num_detections': num_detections,
                       'num_cars': num_cars,
                       'num_bikes': num_bikes})
    stats = df.describe()
    print(stats)
    car_imgs = imgs[car_mask]
    car_imgs.tofile("/export/intel/users/etotoni/cars.dat")

t1 = time.time()
get_stats(AD_images, output_dict['num_detections'], output_dict['detection_classes'],
          output_dict['detection_scores'])
print("stat time", time.time()-t1)

