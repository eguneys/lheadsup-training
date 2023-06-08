import tensorflow as tf
import os
import time
import bisect

from jsnet import JSNet

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

class ApplySqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ApplySqueezeExcitation, self).__init__(**kwargs)

    def build(self, input_dimens):
        self.reshape_size = input_dimens[1][1]

    def call(self, inputs):
        x = inputs[0]
        excited = inputs[1]
        gammas, betas = tf.split(tf.reshape(excited,
            [-1, 1, 1, self.reshape_size]),
            2,
            axis = 3)
        return tf.nn.sigmoid(gammas) * x + betas

class TFProcess:

    def __init__(self, cfg):
        self.cfg = cfg
        self.net = JSNet()
        self.root_dir = os.path.join(self.cfg['training']['path'],
                self.cfg['name'])


        loss_scale = self.cfg['training'].get('loss_scale', 128)
        loss_scale = 1

        self.loss_scale = loss_scale

        self.RESIDUAL_FILTERS = self.cfg['model'].get('filters', 0)
        self.RESIDUAL_BLOCKS = self.cfg['model'].get('residual_blocks', 0)
        self.SE_ratio = self.cfg['model'].get('se_ratio', 0)

        self.value_channels = self.cfg['model'].get('value_channels', 32)

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
       #self.l2reg = tf.keras.regularizers.l2(l=0.5 * (0.0001))
       self.l2reg = tf.keras.regularizers.l2()
       #self.l2reg = None
       input_var = tf.keras.Input(shape=(8, 1, 16))
       outputs = self.construct_net(input_var)

       self.model = tf.keras.Model(inputs=input_var, outputs=outputs)

       self.active_lr = tf.Variable(0.01, trainable=False)
       self.update_lr_manually = False
       self.optimizer = tf.keras.optimizers.legacy.SGD(
               learning_rate=lambda: self.active_lr,
               momentum=0.9,
               nesterov=True)
       self.orig_optimizer = self.optimizer

       try:
           self.aggregator = self.orig_optimizer.aggregate_gradients
       except AttributeError:
            self.aggregator = self.orig_optimizer.gradient_aggregator

       if self.loss_scale != 1:
           self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer, False, self.loss_scale)

       def value_loss(target, output):
           scale = 30.0
           output = tf.cast(output, tf.float32)
           target = target * scale
           output = output * scale

           #tf.print(target, output, tf.reduce_mean(tf.square(target - output)))
           return tf.reduce_mean(tf.square(target - output)) / scale

       def mean_absolute_error(target, output):
           return tf.reduce_mean(tf.abs(target - output))

       def huber_loss(target, output):
           output = tf.cast(output, tf.float32)
           huber = tf.keras.losses.Huber(0.1)
           return tf.reduce_mean(huber(target, output))


       def threshold_loss(target, output):
           difference = tf.abs(target - output)
           loss = tf.where(difference < 0.1, tf.square(difference), 3.0 * difference)
           return tf.reduce_mean(loss)

       #self.value_loss_fn = value_loss
       #self.value_loss_fn = mean_absolute_error
       #self.value_loss_fn = huber_loss
       self.value_loss_fn = threshold_loss

       def accuracy(target, output, threshold=0.09):
           output = tf.cast(output, tf.float32)
           target = tf.cast(target, tf.float32)
           absolute_difference = tf.abs(target - output)
           correct_predictions = tf.less_equal(absolute_difference, threshold)
           accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
           return accuracy

       self.value_accuracy_fn = accuracy


       def value_entropy(target, output):
           target, output = correct_value(target, output)
           softmaxed = tf.nn.softmax(output)

           return tf.math.negative(
                   tf.reduce_mean(
                       tf.reduce_sum(tf.math.xlogy(softmaxed, softmaxed),
                           axis=1)))

       self.value_entropy_fn = value_entropy


       value_loss_w = self.cfg['training']['value_loss_weight']
       reg_term_w = self.cfg['training'].get('reg_term_weight', 1.0)

       def _lossMix(value, _value, reg_term):
           return value_loss_w * value + reg_term_w * reg_term

       self.lossMix = _lossMix

       self.train_metrics = [
               Metric('V', 'Value Loss'),
               Metric('Reg', 'Reg term'),
               Metric('Total', 'Total Loss'),
               ]

       self.test_metrics = [
               Metric('V', 'Value Loss'),
               Metric('V Acc', 'Value Accuracy', suffix='%')]

       self.cfg['training']['lr_boundaries'].sort()
       self.warmup_steps = self.cfg['training'].get('warmup_steps', 0)
       self.lr = self.cfg['training']['lr_values'][0]
       self.test_writer = tf.summary.create_file_writer(
               os.path.join(os.getcwd(),
                   "leelalogs/{}-test".format(self.cfg['name'])))
       self.train_writer = tf.summary.create_file_writer(
               os.path.join(os.getcwd(),
                   "leelalogs/{}-train".format(self.cfg['name'])))



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

    def batch_norm(self, input, name, scale=False):
        #return input
        return tf.keras.layers.BatchNormalization(
                epsilon=1e-5,
                axis=3,
                center=True,
                scale=scale,
                name=name)(input)

    def squeeze_excitation(self, inputs, channels, name):
        assert channels % self.SE_ratio == 0

        pooled = tf.keras.layers.GlobalAveragePooling2D(
                data_format='channels_last')(inputs)
        squeezed = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
                tf.keras.layers.Dense(channels // self.SE_ratio,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=self.l2reg,
                    name=name + '/se/dense1')(pooled))

        excited = tf.keras.layers.Dense(2 * channels,
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                name=name + '/se/dense2')(squeezed)

        return ApplySqueezeExcitation()([inputs, excited])

    def conv_block(self,
            inputs,
            filter_size,
            output_channels,
            name,
            bn_scale=False):
        conv = tf.keras.layers.Conv2D(output_channels,
                filter_size,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_last',
                name=name + '/conv2d')(inputs)
        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
                self.batch_norm(conv, name=name + '/bn', scale=bn_scale))

    def residual_block(self, inputs, channels, name):
        conv1 = tf.keras.layers.Conv2D(channels,
                3,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_last',
                name=name + '/1/conv2d')(inputs)
        out1 = tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
                self.batch_norm(conv1, name + '/1/bn', scale=False))
        conv2 = tf.keras.layers.Conv2D(channels,
                3,
                use_bias=False,
                padding='same',
                kernel_initializer='glorot_normal',
                kernel_regularizer=self.l2reg,
                data_format='channels_last',
                name=name + '/2/conv2d')(out1)

        out2 = self.squeeze_excitation(self.batch_norm(conv2, 
            name + '/2/bn',
            scale=True),
            channels, name=name + '/se')

        return tf.keras.layers.Activation(self.DEFAULT_ACTIVATION)(
                tf.keras.layers.add([inputs, out2]))

    def create_residual_body(self, inputs):
        flow = self.conv_block(inputs,
                filter_size=3,
                output_channels=self.RESIDUAL_FILTERS,
                name='input',
                bn_scale=True)

        for i in range(self.RESIDUAL_BLOCKS):
            flow = self.residual_block(flow,
                self.RESIDUAL_FILTERS,
                name='residual_{}'.format(i + 1))
        return flow


    def construct_net(self, inputs, name=''):
        flow = self.create_residual_body(inputs)

        conv_value = self.conv_block(flow,
            filter_size=1,
            output_channels=32,
            name='value')
        h_conv_value_flat = tf.keras.layers.Flatten()(conv_value)
        h_fc2 = tf.keras.layers.Dense(128,
            kernel_initializer='glorot_normal',
            kernel_regularizer=self.l2reg,
            activation=self.DEFAULT_ACTIVATION,
            name='value/dense1')(h_conv_value_flat)


        h_fc3 = tf.keras.layers.Dense(1,
            kernel_initializer='glorot_normal',
            kernel_regularizer=self.l2reg,
            bias_regularizer=self.l2reg,
            activation='tanh',
            name='value/dense2')(h_fc2)

        outputs = [h_fc3]

        return outputs


    def process_loop(self, batch_size, test_batches, batch_splits=1):
        steps = self.global_step.read_value()
        self.last_steps = steps
        self.time_start = time.time()
        self.profiling_start_step = None

        total_steps = self.cfg['training']['total_steps']

        for _ in range(steps % total_steps, total_steps):
            self.process(batch_size, test_batches, batch_splits=batch_splits)


    def process(self, batch_size, test_batches, batch_splits):
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
            steps = self.train_step(steps, batch_size, batch_splits)

        if steps % self.cfg['training']['test_steps'] == 0 or steps % self.cfg['training']['total_steps'] == 0:
            with tf.profiler.experimental.Trace("Test", step_num=steps):
                self.calculate_test_summaries(test_batches, steps)

        if steps % self.cfg['training']['total_steps'] == 0:
            evaled_steps = steps.numpy()
            self.manager.save(checkpoint_number=evaled_steps)
            print("Model saved in file: {}".format(self.manager.latest_checkpoint))
            path = os.path.join(self.root_dir, self.cfg['name'])
            leela_path = path + "-" + str(evaled_steps)

            #self.net.pb.training_params.training_steps = evaled_steps

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
            x, y = next(self.test_iter)
            metrics = self.calculate_test_summaries_inner_loop(
                    x, y)

            for acc, val in zip(self.test_metrics, metrics):
                acc.accumulate(val)

        #self.net.pb.training_params.learning_rate = self.lr
        #self.net.pb.training_params.accuracy = self.test_metrics[4].get()

        with self.test_writer.as_default():
            for metric in self.test_metrics:
                tf.summary.scalar(metric.long_name, metric.get(), step=steps)
            for w in self.model.weights:
                tf.summary.histogram(w.name, w, step=steps)
        self.test_writer.flush()
            
        print("step {},".format(steps), end='')
        for metric in self.test_metrics:
            print(" {}={:g}{}".format(metric.short_name, metric.get(), metric.suffix),
                    end='')
        print()

    @tf.function()
    def compute_update_ratio(self, before_weights, after_weights, steps):
        deltas = [
                after - before
                for after, before in zip(after_weights, before_weights)
                ]

        delta_norms = [tf.math.reduce_euclidean_norm(d) for d in deltas]
        weight_norms = [
                tf.math.reduce_euclidean_norm(w) for w in before_weights
                ]
        ratios = [(tensor.name, tf.cond(w != 0., lambda: d / w, lambda: -1.))
                for d, w, tensor in zip(delta_norms, weight_norms,
                    self.model.weights)
                if not 'moving' in tensor.name]
        for name, ratio in ratios:
            tf.summary.scalar('update_ratios/' + name, ratio, step=steps)
        ratios = [
                tf.cond(r > 0, lambda: tf.math.log(r) / 2.30258509299,
                    lambda: 200.) for (_, r) in ratios
                ]
        tf.summary.histogram('update_ratios_log10',
                tf.stack(ratios),
                buckets=1000,
                step=steps)

    def save_leelaz_weights(self, filename):
        numpy_weights = []
        for weight in self.model.weights:
            numpy_weights.append([weight.name, weight.numpy()])
        self.net.fill_net_v2(numpy_weights)
        self.net.save_proto(filename)

    @tf.function()
    def calculate_test_summaries_inner_loop(self, x, y):
        outputs = self.model(x, training=False)
        value = outputs
        value_loss = self.value_loss_fn(y, value)
        value_accuracy = self.value_accuracy_fn(y, value)
        
        #tf.print(x, summarize=16)
        #tf.print("Y V", y, value, summarize=10)
        #tf.print("L A", value_loss, value_accuracy)

        metrics = [
                value_loss,
                value_accuracy * 100]

        return metrics

    def apply_grads(self, grads, effective_batch_splits):
        grads = [
                g[0]
                for g in self.aggregator(zip(grads, self.model.trainable_weights))
                ]
        if self.loss_scale != 1:
            grads = self.optimizer.get_unscaled_gradients(grads)
        max_grad_norm = self.cfg['training'].get(
                'max_grad_norm', 10000.0) * effective_batch_splits
        grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads,
            self.model.trainable_weights))
        return grad_norm

    def train_step(self, steps, batch_size, batch_splits):
        if (steps + 1) % self.cfg['training']['train_avg_report_steps'] == 0 or (
                steps + 1) % self.cfg['training']['total_steps'] == 0:
            before_weights = self.read_weights()

        grads = None
        for _ in range(batch_splits):
            x, y = next(self.train_iter)
            metrics, new_grads = self.process_inner_loop(x, y)
            if not grads:
                grads = new_grads
            else:
                grads = self.merge_grads(grads, new_grads)

            for acc, val in zip(self.train_metrics, metrics):
                acc.accumulate(val)

            effective_batch_splits = batch_splits
            self.active_lr.assign(self.lr / effective_batch_splits)
            if self.update_lr_manually:
                self.orig_optimizer.learning_rate = self.active_lr
            grad_norm = self.apply_grads(grads, effective_batch_splits)

            self.global_step.assign_add(1)
            steps = self.global_step.read_value()

            if steps % self.cfg['training']['train_avg_report_steps'] == 0 or steps % self.cfg['training']['total_steps'] == 0:
                time_end = time.time()
                speed = 0
                if self.time_start:
                    elapsed = time_end - self.time_start
                    steps_elapsed = steps - self.last_steps
                    speed = batch_size * (tf.cast(steps_elapsed, tf.float32) /
                            elapsed)
                print("step {}, lr={:g}".format(steps, self.lr), end='')
                for metric in self.train_metrics:
                    print(" {}={:g}{}".format(metric.short_name, metric.get(),
                        metric.suffix),
                        end='')
                print(" ({:g} pos/s)".format(speed))

                after_weights = self.read_weights()
                with self.train_writer.as_default():
                    for metric in self.train_metrics:
                        tf.summary.scalar(metric.long_name,
                                metric.get(),
                                step=steps)
                    tf.summary.scalar("LR", self.lr, step=steps)
                    tf.summary.scalar("Gradient norm",
                            grad_norm / effective_batch_splits,
                            step=steps)
                    self.compute_update_ratio(before_weights, after_weights, steps)
                self.train_writer.flush()

                self.time_start = time_end
                self.last_steps = steps

                for metric in self.train_metrics:
                    metric.reset()
            return steps

    @tf.function()
    def read_weights(self):
        return [w.read_value() for w in self.model.weights]

    @tf.function()
    def process_inner_loop(self, x, y):
        with tf.GradientTape() as tape:
            outputs = self.model(x, training=True)
            value = outputs
            value_loss = self.value_loss_fn(y, value)
            reg_term = sum(self.model.losses)
            total_loss = self.lossMix(value_loss, value_loss, reg_term)
            if self.loss_scale != 1:
                total_loss = self.optimizer.get_scaled_loss(total_loss)
            metrics = [
                    value_loss,
                    reg_term,
                    total_loss,
                    ]

            return metrics, tape.gradient(total_loss, self.model.trainable_weights)

