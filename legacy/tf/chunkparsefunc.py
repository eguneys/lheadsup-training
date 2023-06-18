import tensorflow as tf

def parse_function(planes, prob_rank, prob_card):
    planes = tf.io.decode_raw(planes, tf.float32)
    prob_rank = tf.io.decode_raw(prob_rank, tf.float32)
    prob_card = tf.io.decode_raw(prob_card, tf.float32)

    planes = tf.reshape(planes, (-1, 1, 11, 8))
    prob_rank = tf.reshape(prob_rank, (-1, 9))
    prob_card = tf.reshape(prob_card, (-1, 13))

    return (planes, prob_rank, prob_card)
