import numpy as np
import time
import pandas as pd
import hpat
from hpat import prange, objmode
hpat.multithread_mode = True
import h5py
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


@hpat.jit
def accel_infer():
    #pdir = "/export/intel/users/etotoni/"
    pdir = 'data/'
    #fname = "/export/intel/users/etotoni/BXP5401-data.hdf5"
    fname = 'img2.hdf5'
    f = h5py.File(fname, "r")
    imgs = f["front_cam"][:].astype(np.uint8)
    f.close()

    X = np.fromfile(pdir + "accel_X.dat", np.float64)
    Y = np.fromfile(pdir + "accel_Y.dat", np.float64)
    Z = np.fromfile(pdir + "accel_Z.dat", np.float64)
    T = pd.DatetimeIndex(np.fromfile(pdir + 'accel_T.dat', np.int64))

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z, 'T': T})
    g = 9.81
    df['accel'] = np.sqrt(df.X**2 + df.Y**2 + (df.Z-g)**2)
    df['backward'] = (df.X+df.Y) < 0.0
    threshold = df.accel.mean() + .5 * df.accel.std()
    win_size = 400
    is_brake = (df.rolling(win_size)['accel'].mean() > threshold) & df.backward
    #df = df[(df.rolling('4s', on='T')['accel'].mean() > threshold) & df.backward]
    with objmode():
        run_inference(imgs)
    #imgs.tofile("/export/intel/users/etotoni/accel_cars_v2.dat")
    imgs.tofile("accel_cars_v2.dat")
    return is_brake


# detection
def run_inference(images):
    ROOT_PATH = '/home/etotoni/pse-hpc/python/hpat/nn_examples/mask_rcnn_inception_v2_coco_2018_01_28'
    #ROOT_PATH = '/homes/etotoni/dev/mask_rcnn_inception_v2_coco_2018_01_28'
    PATH_TO_CKPT = ROOT_PATH + '/frozen_inference_graph.pb'

    PATH_TO_LABELS = '/home/etotoni/pse-hpc/python/hpat/nn_examples/models/research/object_detection/data/mscoco_label_map.pbtxt'
    #PATH_TO_LABELS = '/homes/etotoni/python/models/research/object_detection/data/mscoco_label_map.pbtxt'

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
    with detection_graph.as_default():
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

    for i in range(len(images)):
        vis_util.visualize_boxes_and_labels_on_image_array(
            images[i],
            output_dict['detection_boxes'][i],
            output_dict['detection_classes'][i],
            output_dict['detection_scores'][i],
            category_index,
            instance_masks=output_dict['detection_masks'+str(i)],
            use_normalized_coordinates=True,
            line_thickness=8)
    return

if __name__ == '__main__':
    accel_infer()
