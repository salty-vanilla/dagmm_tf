import tensorflow as tf
from tensorflow.python.keras.layers import Input
import numpy as np
import os
import time


class DAGMM:
    def __init__(self, autoencoder,
                 estimation_network,
                 gmm,
                 lambda1=0.1,
                 lambda2=1e-4,
                 learning_rate=1e-4):
        self.autoencoder = autoencoder
        self.estimation_network = estimation_network
        self.gmm = gmm

        with tf.Graph().as_default() as graph:
            self.x = Input(shape=self.autoencoder._input_shape,
                           dtype=tf.float32)

            with tf.name_scope('train'):
                is_training = True
                reuse = False
                self.encoded = self.autoencoder.encode(self.x,
                                                       is_training=is_training,
                                                       reuse=reuse)
                self.decoded = self.autoencoder.decode(self.encoded,
                                                       is_training=is_training,
                                                       reuse=reuse)
                self.distance = self.autoencoder.calc_distance(self.x,
                                                               self.decoded,
                                                               is_training=is_training,
                                                               reuse=reuse)

                self.z = tf.concat([self.encoded, self.distance], axis=-1)

                self.gamma = self.estimation_network(self.z,
                                                     is_training=is_training,
                                                     reuse=reuse)

                self.gmm.build(self.z, self.gamma)
                self.energy = self.gmm.calc_energy(is_training)

            with tf.name_scope('test'):
                is_training = False
                reuse = True
                self._encoded = self.autoencoder.encode(self.x,
                                                        is_training=is_training,
                                                        reuse=reuse)
                self._decoded = self.autoencoder.decode(self._encoded,
                                                        is_training=is_training,
                                                        reuse=reuse)
                self._distance = self.autoencoder.calc_distance(self.x,
                                                                self._decoded,
                                                                is_training=is_training,
                                                                reuse=reuse)

                self._z = tf.concat([self._encoded, self._distance], axis=-1)

                self._gamma = self.estimation_network(self._z,
                                                      is_training=is_training,
                                                      reuse=reuse)

                self._energy = self.gmm.calc_energy(is_training)

            with tf.name_scope('losses'):
                self.mse = tf.reduce_mean((self.x-self.decoded)**2)
                self.energy_loss = tf.reduce_mean(self.energy)
                self.diag_loss = self.gmm.cov_diag_loss()
                self.loss = self.mse + lambda1*self.energy_loss + lambda2*self.diag_loss

            with tf.variable_scope('Summary'):
                self.summary = tf.summary.merge([tf.summary.scalar('mse', self.mse),
                                                 tf.summary.scalar('energy_loss', self.energy_loss),
                                                 tf.summary.scalar('diag_loss', self.diag_loss)])

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.sess = tf.Session(graph=graph)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def fit(self, image_sampler,
            nb_epoch=100,
            save_steps=1,
            logdir='../logs'):
        tb_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        global_step = 0
        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                image_batch = image_sampler()
                if image_batch.shape[0] != batch_size:
                    continue
                _, _, _, _, loss, mse, energy_loss, diag_loss, summary = \
                    self.sess.run([self.optimizer, self.autoencoder.update_ops,
                                   self.estimation_network.update_ops, self.gmm.update_ops,
                                   self.loss, self.mse, self.energy_loss, self.diag_loss,
                                   self.summary],
                                  feed_dict={self.x: image_batch})

                print('iter : {} / {}  {:.1f}[s]  loss : {:.4f}  mse : {:.4f}  energy : {:.4f}  diag : {:.4f}  \r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss, mse, energy_loss, diag_loss), end='')
                tb_writer.add_summary(summary, global_step)
                tb_writer.flush()
                global_step += 1
            if epoch % save_steps == 0:
                self.save(logdir, epoch)

    def save(self, logdir, epoch):
        dst_path = os.path.join(logdir, "epoch_{}.ckpt".format(epoch))
        return self.saver.save(self.sess, save_path=dst_path)

    def restore(self, file_path):
        reader = tf.train.NewCheckpointReader(file_path)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0])
                            for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        var_dict = dict(zip(map(lambda x:
                                x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                current_var = var_dict[saved_var_name]
                var_shape = current_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(current_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, file_path)

    def predict_energy(self, x,
                       batch_size=32):
        outputs = np.empty((0, ), dtype=np.float32)
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_energy_on_batch(x_batch)
            outputs = np.append(outputs, o)
        return outputs

    def predict_energy_on_batch(self, x):
        return self.sess.run(self._energy,
                             feed_dict={self.x: x})