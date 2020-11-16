from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os


def create_op(func, **placeholders):
    op = func(**placeholders)

    def f(**kwargs):
        feed_dict = {}
        for argname, argvalue in kwargs.items():
            placeholder = placeholders[argname]
            feed_dict[placeholder] = argvalue
        return tf.get_default_session().run(op, feed_dict=feed_dict)

    return f

downscale = create_op(
    func=tf.image.resize_images,
    images=tf.placeholder(tf.float32, [None, None, None]),
    size=tf.placeholder(tf.int32, [2]),
    method=tf.image.ResizeMethod.AREA,
)

upscale = create_op(
    func=tf.image.resize_images,
    images=tf.placeholder(tf.float32, [None, None, None]),
    size=tf.placeholder(tf.int32, [2]),
    method=tf.image.ResizeMethod.BICUBIC,
)

decode_jpeg = create_op(
    func=tf.image.decode_jpeg,
    contents=tf.placeholder(tf.string),
)

decode_png = create_op(
    func=tf.image.decode_png,
    contents=tf.placeholder(tf.string),
)

rgb_to_grayscale = create_op(
    func=tf.image.rgb_to_grayscale,
    images=tf.placeholder(tf.float32),
)

grayscale_to_rgb = create_op(
    func=tf.image.grayscale_to_rgb,
    images=tf.placeholder(tf.float32),
)

encode_jpeg = create_op(
    func=tf.image.encode_jpeg,
    image=tf.placeholder(tf.uint8),
)

encode_png = create_op(
    func=tf.image.encode_png,
    image=tf.placeholder(tf.uint8),
)

crop = create_op(
    func=tf.image.crop_to_bounding_box,
    image=tf.placeholder(tf.float32),
    offset_height=tf.placeholder(tf.int32, []),
    offset_width=tf.placeholder(tf.int32, []),
    target_height=tf.placeholder(tf.int32, []),
    target_width=tf.placeholder(tf.int32, []),
)

pad = create_op(
    func=tf.image.pad_to_bounding_box,
    image=tf.placeholder(tf.float32),
    offset_height=tf.placeholder(tf.int32, []),
    offset_width=tf.placeholder(tf.int32, []),
    target_height=tf.placeholder(tf.int32, []),
    target_width=tf.placeholder(tf.int32, []),
)

to_uint8 = create_op(
    func=tf.image.convert_image_dtype,
    image=tf.placeholder(tf.float32),
    dtype=tf.uint8,
    saturate=True,
)

to_float32 = create_op(
    func=tf.image.convert_image_dtype,
    image=tf.placeholder(tf.uint8),
    dtype=tf.float32,
)


def load(path):
    with open(path, "rb") as f:
        contents = f.read()
        
    _, ext = os.path.splitext(path.lower())

    if ext == ".jpg":
        image = decode_jpeg(contents=contents)
    elif ext == ".png":
        image = decode_png(contents=contents)
    else:
        raise Exception("invalid image suffix")

    return to_float32(image=image)


def find(d):
    result = []
    for filename in os.listdir(d):
        _, ext = os.path.splitext(filename.lower())
        if ext == ".jpg" or ext == ".png":
            result.append(os.path.join(d, filename))
    result.sort()
    return result


def save(image, path, replace=False):
    _, ext = os.path.splitext(path.lower())
    image = to_uint8(image=image)
    if ext == ".jpg":
        encoded = encode_jpeg(image=image)
    elif ext == ".png":
        encoded = encode_png(image=image)
    else:
        raise Exception("invalid image suffix")

    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    if os.path.exists(path):
        if replace:
            os.remove(path)
        else:
            raise Exception("file already exists at " + path)

    with open(path, "wb") as f:
        f.write(encoded)
