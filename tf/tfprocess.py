import tensorflow as tf
import os
import time

class Metric:

    def __init__(self, short_name, long_name, suffix=''):
        self.short_name = short_name
        self.long_name = long_name
        self.suffix = suffix
        self.value = 0.0
        self.count = 0

    def assign(self, value):
        self.value = value
        self.count = 1

    def accumulate(self, value):
        if self.count > 0:
            self.value = self.value + value
            self.count = self.count + 1
        else:
            self.assign(value)

    def get(self):
        if self.count == 0:
            return self.value
        return self.value / self.count

    def reset(self):
        self.value = 0.0
        self.count = 0

class TFProcess:

    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = os.path.join(self.cfg['training']['path'],
                self.cfg['name'])

        self.RESIDUAL_FILTERS = self.cfg['model'].get('filters', 0)
        self.RESIDUAL_BLOCKS = self.cfg['model'].get('residual_blocks', 0)

        self.rank_channels = self.cfg['model'].get('rank_channels', 32)
        self.card_channels = self.cfg['model'].get('card_channels', 32)

        self.DEFAULT_ACTIVATION = 'relu'

        self.global_step = tf.Variable(0,
                name='global_step',
                trainable=False,
                dtype=tf.int64)


    def init(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.train_iter = iter(self.train_dataset)

        self.test_dataset = test_dataset
        self.test_iter = iter(self.test_dataset)

        self.init_net()

    def init_net(self):
       self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
       input_var = tf.keras.Input(shape=(1, 11, 8))
       outputs = self.construct_net(input_var)

       self.model = tf.keras.Model(inputs=input_var, outputs=outputs)

       self.active_lr = tf.Variable(0.01, trainable=False)
       self.optimizer = tf.keras.optimizers.legacy.SGD(
               learning_rate=lambda: self.active_lr,
               momentum=0.9,
               nesterov=True)
       self.orig_optimizer = self.optimizer


       def correct_rank(target, output):
           output = tf.cast(output, tf.float32)

           target = tf.nn.relu(target)
           return target, output

       def rank_loss(target, output):
           target, output = correct_rank(target, output)
           rank_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                   labels=tf.stop_gradient(target), logits=output)
           return tf.reduce_mean(input_tensor=rank_cross_entropy)

       self.rank_loss_fn = rank_loss

       def rank_accuracy(target, output):
           target, output = correct_rank(target, output)
           return tf.reduce_mean(
                   tf.cast(
                       tf.equal(tf.argmax(input=target, axis=1),
                           tf.argmax(input=output, axis=1)), tf.float32))

       self.rank_accuracy_fn = rank_accuracy


       def rank_entropy(target, output):
           target, output = correct_rank(target, output)
           softmaxed = tf.nn.softmax(output)

           return tf.math.negative(
                   tf.reduce_mean(
                       tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                           axis=1)))

       self.rank_entropy_fn = rank_entropy


       def rank_uniform_loss(target, output):
           uniform = tf.where(tf.greater_equal(target, 0),
                   tf.ones_like(target), tf.zeros_like(target))

           balanced_uniform = uniform / tf.reduce_sum(
                   uniform, axis=1, keepdims=True)

           target, output = correct_rank(target, output)

           rank_cross_entropy = \
                   tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(balanced_uniform),
                           logits=output)

           return tf.reduce_mean(input_tensor=rank_cross_entropy)

       self.rank_uniform_loss_fn = rank_uniform_loss


       def correct_card(target, output):
           output = tf.cast(output, tf.float32)

           target = tf.nn.relu(target)
           return target, output



       def card_loss(target, output):
           target, output = correct_card(target, output)
           card_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                   labels=tf.stop_gradient(target), logits=output)
           return tf.reduce_mean(input_tensor=card_cross_entropy)

       self.card_loss_fn = card_loss

       def card_accuracy(target, output):
           target, output = correct_card(target, output)
           return tf.reduce_mean(
                   tf.cast(
                       tf.equal(tf.argmax(input=target, axis=1),
                           tf.argmax(input=output, axis=1)), tf.float32))

       self.card_accuracy_fn = card_accuracy




       self.test_metrics = [
               Metric('P', 'Policy Loss')
               ]

       self.cfg['training']['lr_boundaries'].sort()
       self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
       self.lr = self.cfg['training']['lr_values'][0]

       self.checkpoint = tf.train.Checkpoint(optimizer=self.orig_optimizer,
               model=self.model,
               global_step=self.global_step)

       self.manager = tf.train.CheckpointManager(
               self.checkpoint,
               directory=self.root_dir,
               max_to_keep=50,
               keep_checkpoint_every_n_hours=24,
               checkpoint_name=self.cfg['name'])


    def restore(self):
        if self.manager.latest_checkpoint is not None:
            print("Restoring from {0}".format(self.manager.latest_checkpoint))
            self.checkpoint.restore(self.manager.latest_checkpoint)

    def conv_block(self,
            inputs,
            filter_size,
            output_channels,
            name):
        conv = tf.keras.layers.Conv2D(output_channels,
                filter_size,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_last',
                name=name + '/conv2d')(inputs)
        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(conv)

    def residual_block(self, inputs, channels, name):
        conv1 = tf.keras.layers.Conv2D(channels,
                3,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_last',
                name=name + '/1/conv2d')(inputs)
        out1 = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(conv1)
        conv2 = tf.keras.layers.Conv2D(channels,
                3,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_last',
                name=name + '/2/conv2d')(out1)

        out2 = conv2

        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
                tf.keras.layers.add([inputs, out2]))

    def create_residual_body(self, inputs):
        flow = self.conv_block(inputs,
                filter_size=3,
                output_channels=self.RESIDUAL_FILTERS,
                name='input')

        for i in range(self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow,
                self.RESIDUAL_FILTERS,
                name='residual_{}'.format(i + 1))
        return flow


    def construct_net(self, inputs, name=''):
        flow = self.create_residual_body(inputs)

        conv_rank = self.conv_block(flow,
            filter_size=1,
            output_channels=self.rank_channels,
            name='rank')
        h_conv_rank_flat = tf.keras.layers.Flatten()(conv_rank)
        h_fc1 = tf.keras.layers.Dense(9,
            kernel_initializer='glorot_normal',
            kernel_regularizer=self.l2reg,
            bias_regularizer=self.l2reg,
            name='rank/dense')(h_conv_rank_flat)


        conv_card = self.conv_block(flow,
            filter_size=1,
            output_channels=self.card_channels,
            name='card')
        h_conv_card_flat = tf.keras.layers.Flatten()(conv_card)
        h_fc3 = tf.keras.layers.Dense(13,
            kernel_initializer='glorot_normal',
            kernel_regularizer=self.l2reg,
            bias_regularizer=self.l2reg,
            name='card/dense')(h_conv_card_flat)

        outputs = [h_fc1, h_fc3]

        return outputs


    def process_loop(self, batch_size, test_batches):
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg['training']['total_steps']

        for _ in range(steps % total_steps, total_steps):
            self.process(batch_size, test_batches)


    def process(self, batch_size, test_batches):
        steps = self.global_step.read_value()

        if steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps + 1):
                self.calculate_test_summaries(test_batches, steps + 1)


        lr_values = self.cfg['training']['lr_values']
        lr_boundaries = self.cfg['training']['lr_boundaries']
        steps_total = steps % self.cfg['training']['total_steps']

        self.lr = lr_values[bisect.bisect_right(lr_boundaries, steps_total)]
        if self.warmup_steps > 0 and steps < self.warmup_steps:
            self.lr = self.lr * tf.cast(steps + 1,
                    tf.float32) / self.warmup_steps


        with tf.profiler.experimental.Trace("Train", step_num=steps):
            steps = self.train_step(steps)

        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries(test_batches, steps)

        if steps % self.cfg['training']['total_steps'] == 0:
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(self.manager.latest_checkpoint))
            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(evaled_steps)

            self.net.pb.training_params.training_steps = evaled_steps

            self.save_leelaz_weights(leela_path)

        if self.profiling_start_step is not None and (
                steps >= self.profiling_start_step +
                steps % self.cfg['training']['total_steps'] == 0):
            tf.profiler.experimental.stop()
            self.profiling_start_step = None


    def calculate_test_summaries(self, test_batches, steps):
        for metric in self.test_metrics:
            metric.reset()
        for _ in range(0, test_batches):
            x, y, z = next(self.test_iter)
            metrics = self.calculate_test_summaries_inner_loop(
                    x, y, z)

            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)

        self.net.pb.training_params.leraning_rate = self.lr
        self.net.pb.training_params.accuracy = self.test_metrics[4].get()

        with self.test_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
            for w in self.model.weights:
                tf.summary.histogram(w.name, w, step=steps)
        self.test_writer.flush()
            
        print("step {},".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(), metrix.suffix),
                    end='')
        print()


    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y, z):
        outputs = self.model(x, training=False)
        rank = outputs[0]
        card = outputs[1]
        rank_loss = self.rank_loss_fn(y, rank)
        rank_accuracy = self.rank_accuracy_fn(y, rank)
        rank_entropy = self.rank_entropy_fn(y, rank)
        rank_ul = self.rank_uniform_loss_fn(y, rank)

        card_loss = self.card_loss_fn(z, card)
        card_accuracy = self.card_accuracy_fn(z, card)
        #card_entropy = self.card_entropy_fn(z, card)
        #card_ul = self.card_uniform_loss_fn(z, card)

        metrics = [
                rank_loss,
                card_loss,
                rank_accuracy * 100,
                card_accuracy * 100,
                rank_entropy,
                rank_ul]

        return metrics
