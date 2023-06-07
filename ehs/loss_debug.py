#!/usr/bin/env python3

import tensorflow as tf

def value_loss(target, output):
    scale = 30.0
    output = tf.cast(output, tf.float32)
    target = target * scale
    output = output * scale

    return tf.reduce_mean(tf.square(target - output)) / scale

def mean_absolute_error(target, output):
    return tf.reduce_mean(tf.abs(target - output))

def huber_loss(target, output):
    output = tf.cast(output, tf.float32)
    huber = tf.keras.losses.Huber(0.1)
    return tf.reduce_mean(huber(target, output))


def compare(a, b):
    vl = value_loss(a, b)
    mae = mean_absolute_error(a, b)
    hl = huber_loss(a, b)

    tf.print(a, b)
    print("VL MAE HL")
    tf.print(vl, mae, hl)



bs = [
        [0.1, 0.1, 0.1],
        [0.1, 0.12, 0.1],
        [0.1, 0.12, 0.12],
        [0.12, 0.12, 0.12]
        ]

bs = [
        [0.5],
        [0.51],
        [0.52],
        [0.55],
        [0.59],
        [0.6],
        [0.7],
        [0.8]
        ]

a = tf.constant([0.5])

[compare(a, tf.constant(b)) for b in bs]

