import tensorflow as tf

def accuracy(target, output, threshold=0.07):
    output = tf.cast(output, tf.float32)
    target = tf.cast(target, tf.float32)
    absolute_difference = tf.abs(target - output)
    correct_predictions = tf.less_equal(absolute_difference, threshold)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy 
