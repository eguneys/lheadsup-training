import tensorflow as tf

def parse_function(cards, value):
    cards = tf.io.decode_raw(cards, tf.float32)
    value = tf.io.decode_raw(value, tf.float32)

    cards = tf.reshape(cards, (-1, 1, 15, 8))
    value = tf.reshape(value, (-1, 1))

    return (cards, value)
