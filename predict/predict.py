# restore
# input: image,
# output: box and mask with visualization

import os, sys
import tensorflow as tf
import time
from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import libs.configs.config_v1 as cfg
import libs.datasets.dataset_factory as datasets
import libs.nets.nets_factory as network

import libs.nets.pyramid_network as pyramid_network
from libs.visualization.pil_utils import cat_id_to_cls_name, draw_img, draw_bbox, draw_mask

FLAGS = tf.app.flags.FLAGS

# data
image, ih, iw, gt_boxes, gt_masks, num_instances, img_id = \
    datasets.get_dataset(FLAGS.dataset_name,
                         FLAGS.dataset_split_name,
                         FLAGS.dataset_dir,
                         FLAGS.im_batch,
                         is_training=True)

data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
        dtypes=(
            image.dtype, ih.dtype, iw.dtype,
            gt_boxes.dtype, gt_masks.dtype,
            num_instances.dtype, img_id.dtype))
enqueue_op = data_queue.enqueue((image, ih, iw, gt_boxes, gt_masks, num_instances, img_id))
data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
(image, ih, iw, gt_boxes, gt_masks, num_instances, img_id) =  data_queue.dequeue()
im_shape = tf.shape(image)
image = tf.reshape(image, (im_shape[0], im_shape[1], im_shape[2], 3))


# network
logits, end_points, pyramid_map = network.get_network(FLAGS.network, image,
        weight_decay=FLAGS.weight_decay, is_training=True)
outputs = pyramid_network.build(end_points, im_shape[1], im_shape[2], pyramid_map,
        num_classes=81,
        base_anchors=9,
        is_training=True,
        gt_boxes=gt_boxes, gt_masks=gt_masks,
        loss_weights=[0.2, 0.2, 1.0, 0.2, 1.0])

total_loss = outputs['total_loss']
losses  = outputs['losses']
batch_info = outputs['batch_info']
regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

input_image = end_points['input']
final_box = outputs['final_boxes']['box']
final_cls = outputs['final_boxes']['cls']
final_prob = outputs['final_boxes']['prob']
final_gt_cls = outputs['final_boxes']['gt_cls']
gt = outputs['gt']

#############################
tmp_0 = outputs['losses']
tmp_1 = outputs['losses']
tmp_2 = outputs['losses']
tmp_3 = outputs['losses']
tmp_4 = outputs['losses']

# tmp_0 = outputs['tmp_0']
# tmp_1 = outputs['tmp_1']
# tmp_2 = outputs['tmp_2']
tmp_3 = outputs['tmp_3']
tmp_4 = outputs['tmp_4']
final_mask = outputs['mask']['mask']
############################

# initialization
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
        )
sess.run(init_op)

# restore
checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
restorer = tf.train.Saver()
restorer.restore(sess, checkpoint_path)
print ('restored previous model %s from %s'\
        %(checkpoint_path, FLAGS.train_dir))

# predict
## main loop
coord = tf.train.Coordinator()
threads = []
# print (tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                     start=True))

tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(10):

    start_time = time.time()

    tot_loss, reg_lossnp, img_id_str, \
    rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss, mask_loss, \
    gt_boxesnp, \
    final_masknp, \
    gt_masksnp, \
    rpn_batch_pos, rpn_batch, refine_batch_pos, refine_batch, mask_batch_pos, mask_batch, \
    input_imagenp, final_boxnp, final_clsnp, final_probnp, final_gt_clsnp, gtnp, tmp_0np, tmp_1np, tmp_2np, tmp_3np, tmp_4np= \
                 sess.run([total_loss, regular_loss, img_id] +
                          losses +
                          [gt_boxes] +
                          [final_mask] +
                          [gt_masks] +
                          batch_info +
                          [input_image] + [final_box] + [final_cls] + [final_prob] + [final_gt_cls] + [gt] + [tmp_0] + [tmp_1] + [tmp_2] + [tmp_3] + [tmp_4])

    duration_time = time.time() - start_time
    if step % 1 == 0:
        print ( """iter %d: image-id:%07d, time:%.3f(sec), regular_loss: %.6f, """
                """total-loss %.4f(%.4f, %.4f, %.6f, %.4f, %.4f), """
                """instances: %d, """
                """batch:(%d|%d, %d|%d, %d|%d)"""
               % (step, img_id_str, duration_time, reg_lossnp,
                  tot_loss, rpn_box_loss, rpn_cls_loss, refined_box_loss, refined_cls_loss, mask_loss,
                  gt_boxesnp.shape[0],
                  rpn_batch_pos, rpn_batch, refine_batch_pos, refine_batch, mask_batch_pos, mask_batch))

        img_predict, _ = draw_mask(step,
                  np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0),
                  name='est',
                  bbox=final_boxnp,
                  mask=final_masknp,
                  label=final_clsnp,
                  prob=final_probnp,
                  gt_label=np.argmax(np.asarray(final_gt_clsnp),axis=1),
                  )

        img_gt, _ = draw_mask(step,
                  np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0),
                  name='gt',
                  bbox=gtnp[:,0:4],
                  mask=gt_masksnp,
                  label=np.asarray(gtnp[:,4], dtype=np.uint8),
                  )

        fig = plt.figure()
        plot_predict = fig.add_subplot(1,2,1)
        plot_predict.set_title("Predicted")
        plt.imshow(img_predict)

        plot_gt = fig.add_subplot(1,2,2)
        plot_gt.set_title("Ground truth")
        plt.imshow(img_gt)

        plt.show()

        # draw_bbox(step,
        #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0),
        #           name='est',
        #           bbox=final_boxnp,
        #           label=final_clsnp,
        #           prob=final_probnp,
        #           gt_label=np.argmax(np.asarray(final_gt_clsnp),axis=1),
        #           )

        # draw_bbox(step,
        #           np.uint8((np.array(input_imagenp[0])/2.0+0.5)*255.0),
        #           name='gt',
        #           bbox=gtnp[:,0:4],
        #           label=np.asarray(gtnp[:,4], dtype=np.uint8),
        #           )

        print ("labels")
        # print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(final_gt_clsnp),axis=1)))[1:])
        # print (cat_id_to_cls_name(np.unique(np.asarray(gt_boxesnp, dtype=np.uint8)[:,4])))
        print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(tmp_3np),axis=1)))[1:])
        #print (cat_id_to_cls_name(np.unique(np.argmax(np.asarray(gt_boxesnp)[:,4],axis=1))))
        print ("classes")
        print (cat_id_to_cls_name(np.unique(np.argmax(np.array(tmp_4np),axis=1))))
        # print (np.asanyarray(tmp_3np))

        #print ("ordered rois")
        #print (np.asarray(tmp_0np)[0])
        #print ("pyramid_feature")
        #print ()
         #print(np.unique(np.argmax(np.array(final_probnp),axis=1)))
        #for var, val in zip(tmp_2, tmp_2np):
        #    print(var.name)
        #print(np.argmax(np.array(tmp_0np),axis=1))


        if np.isnan(tot_loss) or np.isinf(tot_loss):
            print (gt_boxesnp)
            raise
