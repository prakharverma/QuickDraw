import abc

import numpy as np
import tensorflow as tf
import gpflow

import log_file


class BaseModel:
    def __init__(
        self, x, y, n_classes, n_inducing_pnts, lr, output_path, inducing_pnts
    ):
        # Load inducing points which are already fixed
        if inducing_pnts[0] is not None and inducing_pnts[1] is not None:
            self.random_inducing_pnts = inducing_pnts[0]
            self.y_ind = inducing_pnts[1]
        else:
            inducing_pnt_loc = np.random.randint(0, x.shape[0], n_inducing_pnts)
            self.random_inducing_pnts = x[inducing_pnt_loc]
            self.y_ind = y[inducing_pnt_loc]

        self.log_file = log_file.LogFile(output_path)
        self.n_classes = n_classes
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)

        if n_classes == 2:
            self.num_latent_gps = 1
            self.liklihood = gpflow.likelihoods.Bernoulli()
        else:
            self.num_latent_gps = n_classes
            self.liklihood = gpflow.likelihoods.Softmax(n_classes)

        self.model = None
        self.output_path = output_path

    @abc.abstractmethod
    def train(self, x, y, epoch, minibatch_size, x_test, y_test):
        raise NotImplementedError

    @abc.abstractmethod
    def train_iter(
        self, x, y, max_iteration, minibatch_size, x_test, y_test, print_interval
    ):
        raise NotImplementedError

    def create_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(
            epoch=tf.Variable(1), optim=self.optimizer, model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt, self.output_path, max_to_keep=2
        )
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    def save_checkpoint(self):
        self.ckpt.epoch.assign_add(1)
        self.manager.save()

    def test(self, x, y, minibatch_size=128):
        if x is None or y is None:
            return 0.0

        test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(x.shape[0])
        test_iter = iter(test_dataset.batch(minibatch_size))

        if self.n_classes:
            acc = []
            for batch_x, batch_y in test_iter:
                acc.append(np.mean((self.model.predict_y(batch_x)[0] > 0.5).numpy().astype("float") == batch_y))
        else:
            acc = []
            for batch_x, batch_y in test_iter:
                pred = self.model.predict_y(batch_x)[0]
                pred_argmax = tf.reshape(tf.argmax(pred, axis=1), (-1, 1))
                acc.append(np.mean(pred_argmax == batch_y))

        return np.mean(acc)

    def nlpd(self, x, y, minibatch_size=128):
        if x is None or y is None:
            return 0.0

        test_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(x.shape[0])
        train_iter = iter(test_dataset.batch(minibatch_size))
        nlpd = []
        for batch_x, batch_y in train_iter:
            nlpd_val = -tf.reduce_mean(
                self.model.predict_log_density((batch_x, batch_y))
            )
            nlpd.append(nlpd_val)
        return np.mean(nlpd)

    def get_model(self):
        return self.model

    def log_data(self, elbo, nlpd, acc):
        self.log_file.record(f"ELBO: Epoch {int(self.ckpt.epoch)} : {elbo}")
        self.log_file.record(f"NLPD: Epoch {int(self.ckpt.epoch)} : {nlpd}")
        self.log_file.record(f"ACC: Epoch {int(self.ckpt.epoch)} : {acc}")

    def print_summary(self):
        gpflow.utilities.print_summary(self.model)
