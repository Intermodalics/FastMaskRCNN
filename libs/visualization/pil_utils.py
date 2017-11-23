import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from scipy.misc import imresize
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
_DEBUG = True

def draw_img(step, image, name='', image_height=1, image_width=1, rois=None):
    #print("image")
    #print(image)
    #norm_image = np.uint8(image/np.max(np.abs(image))*255.0)
    norm_image = np.uint8(image/0.1*127.0 + 127.0)
    #print("norm_image")
    #print(norm_image)
    source_img = Image.fromarray(norm_image)
    return source_img.save(FLAGS.train_dir + 'test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

def draw_rectangle(draw, box, color, width=1):
    for i in range(width):
        rect_start = (box[0] - i, box[1] - i)
        rect_end = (box[2] + i, box[3] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

def draw_bbox(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, prob=None):
    #print(prob[:,label])
    source_img = Image.fromarray(image)
    b, g, r = source_img.split()
    source_img = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(source_img)
    color_good = 'blue'
    color = color_good
    color_intermediate = 'red'
    color_background = 'yellow'
    color_mismatch = 'green'
    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None:
                if prob is not None:
                    if (prob[i,label[i]] > 0.5) and (label[i] > 0):
                    #if (prob[i,label[i]] > 0.0):
                        if gt_label is not None:
                            #text  = cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i])
                            if label[i] != gt_label[i]:
                                #color = '#ff0000'#draw.text((2+bbox[i,0], 2+bbox[i,1]), cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]), fill='#ff0000')
                                color = color_mismatch
                            else:
                                color = color_good
                                #color = '#0000ff'
                        else:
                            color = color_mismatch
                            #text = cat_id_to_cls_name(label[i])
                        #draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                        #if _DEBUG is True:
                        #    print("plot",label[i], prob[i,label[i]])
                        #draw.rectangle(box,fill=None,outline=color)
                    else:
                        if (prob[i,label[i]] <= 0.5):
                            color = color_intermediate
                        else:
                            color = color_background
                        #if _DEBUG is True:
                        #    print("skip",label[i], prob[i,label[i]])
                    print("plot",label[i], prob[i,label[i]])
                else:
                    color = color_good
                    #text = cat_id_to_cls_name(label[i])
                    #draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                draw_rectangle(draw, box,fill=None,outline=color)


    return source_img.save(FLAGS.train_dir + '/est_imgs/test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

def draw_mask(step, image, name='', image_height=1, image_width=1, bbox=None, mask=None, label=None, gt_label=None, prob=None):
    #print(prob[:,label])
    source_img = Image.fromarray(image)
    b, g, r = source_img.split()
    source_img = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(source_img)
    print ('bbox', bbox)
    print ('label', label)
    print ('prob', prob)

    color_top = 'blue'
    color_occluded = 'red'
    color_background = 'yellow'
    mask_color_id = 0

    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None:
                # mask_color_id = np.random.randint(15)
                box1 = np.floor(box).astype('uint16')
                box_w = box1[2]-box1[0]
                box_h = box1[3]-box1[1]

                prob_box = prob[i, label[i]]
                if label[i] == 0:
                    color = color_background
                elif label[i] == 1:
                    color = color_top
                elif label[i] == 2:
                    color = color_occluded
                # if prob_box < 0.7:
                    # color = color_background
                if (color != color_background):
                    m = np.array(mask * 255.0)
                    m = np.transpose(m,(0,3,1,2))
                    mask_color_id += 1
                    print mask_color_id
                    color_img = color_id_to_color_code((mask_color_id)%15) * np.ones((box_h,box_w,1)) * 255
                    color_img = Image.fromarray(color_img.astype('uint8')).convert('RGBA')
                    # color_img = Image.fromarray(color_img.astype('uint8'))
                    #color_img = Image.new("RGBA", (bbox_w,bbox_h), np.random.rand(1,3) * 255 )
                    # print(bbox_w, bbox_h, i, label[i], bbox.shape)
                    resized_m = imresize(m[i][label[i]], [box_h, box_w], interp='bilinear') #label[i]
                    resized_m[resized_m >= 128] = 128
                    resized_m[resized_m < 128] = 0
                    resized_m = Image.fromarray(resized_m.astype('uint8'), 'L')

                    print('label: ', label[i])
                    print('prob: ', prob_box)
                    #print(resized_m)
                    width = int((prob_box-0.5)*8)
                    if width > 0:
                        source_img.paste(color_img , (box1[0], box1[1]), mask=resized_m)
                        draw_rectangle(draw, box1, color=color, width = width)
    return source_img, source_img.save(FLAGS.train_dir + '/est_imgs/test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

def cat_id_to_cls_name(catId):
    cls_name = np.array([  'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    return cls_name[catId]

def color_id_to_color_code(colorId):
    color_code = np.array([[0, 169, 252],
                           [178, 31, 53],
                           [216, 39, 53],
                           [104, 30, 126],
                           [125, 60, 181],
                           [255, 255, 53],
                           [0, 117, 58],
                           [0, 158, 71],
                           [22, 221, 53],
                           [255, 116, 53],
                           [0, 82, 165],
                           [0, 121, 231],
                           [255, 161, 53],
                           [255, 203, 53],
                           [189, 122, 246]])
    return color_code[colorId]
