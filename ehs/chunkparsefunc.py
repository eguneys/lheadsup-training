import tensorflow as tf

def parse_function(cards, hs, ppot, npot):
    cards = tf.io.decode_raw(cards, tf.float32)
    hs = tf.io.decode_raw(hs, tf.float32)
    ppot = tf.io.decode_raw(ppot, tf.float32)
    npot = tf.io.decode_raw(npot, tf.float32)


    cards = tf.reshape(cards, (-1, 16, 8, 1))
    hs = tf.reshape(hs, (-1, 1))
    ppot = tf.reshape(ppot, (-1, 1))
    npot = tf.reshape(npot, (-1, 1))

    # channels_last
    cards = tf.transpose(cards, perm=[0, 2, 3, 1])

    return (cards, hs, ppot, npot)
