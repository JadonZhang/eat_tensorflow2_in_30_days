
import tensorflow as tf
import os
import numpy as np


# inpath：原始数据路径 outpath:TFRecord文件输出路径
def create_tfrecords(inpath, outpath):
    writer = tf.io.TFRecordWriter(outpath)
    dirs = os.listdir(inpath)
    for index, name in enumerate(dirs):
        class_path = inpath + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            # img = tf.image.decode_image(img)
            # img = tf.image.encode_jpeg(img) #统一成jpeg格式压缩
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
                }))
            writer.write(example.SerializeToString())
    writer.close()


create_tfrecords("./data/cifar2/test/", "./data/cifar2_test.tfrecords/")