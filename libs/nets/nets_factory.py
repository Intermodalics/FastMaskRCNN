from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from . import resnet_v1
from .resnet_v1 import resnet_v1_50 as resnet50
from .resnet_utils import resnet_arg_scope
from .resnet_v1 import resnet_v1_101 as resnet101

from . import vgg
from .vgg import vgg_16 as  vgg16

from . import mobilenet_v1
from .mobilenet_v1 import mobilenet_v1_base as mobilenet

slim = tf.contrib.slim

pyramid_maps = {
  'resnet50': {'C1':'resnet_v1_50/conv1/Relu:0',
               'C2':'resnet_v1_50/block1/unit_2/bottleneck_v1',
               'C3':'resnet_v1_50/block2/unit_3/bottleneck_v1',
               'C4':'resnet_v1_50/block3/unit_5/bottleneck_v1',
               'C5':'resnet_v1_50/block4/unit_3/bottleneck_v1',
               },
  'resnet101': {'C1': '', 'C2': '',
                'C3': '', 'C4': '',
                'C5': '',
               },
  'vgg16': {'C1': 'vgg_16/conv1/conv1_2',
            'C2': 'vgg_16/conv2/conv2_2',
            'C3': 'vgg_16/conv3/conv3_3',
            'C4': 'vgg_16/conv4/conv4_3',
            'C5': 'vgg_16/conv5/conv5_3',
            },
  'mobilenet': {'C1': 'Conv2d_0',
                'C2': 'Conv2d_1_pointwise',
                'C3': 'Conv2d_5_pointwise',
                'C4': 'Conv2d_9_pointwise',
                'C5': 'Conv2d_13_pointwise',
                }

}

def get_network(name, image, weight_decay=0.000005, is_training=False):

    if name == 'resnet50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet50(image, 1000, is_training=is_training)

    if name == 'resnet101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
            logits, end_points = resnet50(image, 1000, is_training=is_training)

    if name == 'resnext50':
        name
    if name == 'vgg16':
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
            logits, end_points = vgg16(image, 1000, is_training=is_training)

    if name == 'mobilenet':
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)):
            logits, end_points = mobilenet(image)

    end_points['input'] = image
    return logits, end_points, pyramid_maps[name]
