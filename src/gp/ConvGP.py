import math
import gpflow
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from gpflow import set_trainable

from BaseModel import BaseModel

import sys
sys.path.append("..")
import utils


class ConvGP(BaseModel):
    def __init__(
        self,
        x,
        y,
        n_patches=1000,
        n_classes=2,
        patch_shape=(5, 5),
        lr=0.001,
        output_model_path="model_conv/",
        inducing_pnts=(None, None),
    ):

        super(ConvGP, self).__init__(
            x, y, n_classes, n_patches, lr, output_model_path, inducing_pnts
        )

        self._init_lambda()
        image_l = int(math.sqrt(x.shape[1]))

        conv_k = gpflow.kernels.Convolutional(
            gpflow.kernels.SquaredExponential(), (image_l, image_l), patch_shape
        )
        conv_k.base_kernel.lengthscales = gpflow.Parameter(
            1.0, transform=self.positive_with_min()
        )
        # Weight scale and variance are non-identifiable. We also need to prevent variance from shooting off crazily.
        conv_k.base_kernel.variance = gpflow.Parameter(
            1.0, transform=self.constrained()
        )
        conv_k.weights = gpflow.Parameter(
            conv_k.weights.numpy(), transform=self.max_abs_1()
        )

        patches = utils.sample_patches(conv_k, x, patch_shape, n_patches)
        conv_f = gpflow.inducing_variables.InducingPatches(patches)

        self.model = gpflow.models.SVGP(
            conv_k,
            self.liklihood,
            conv_f,
            num_latent_gps=self.num_latent_gps,
            num_data=x.shape[0],
        )

        self.create_checkpoint()

    def _init_lambda(self):
        f64 = lambda x: np.array(x, dtype=np.float64)
        self.positive_with_min = lambda: tfp.bijectors.AffineScalar(shift=f64(1e-4))(
            tfp.bijectors.Softplus()
        )
        self.constrained = lambda: tfp.bijectors.AffineScalar(
            shift=f64(1e-4), scale=f64(100.0)
        )(tfp.bijectors.Sigmoid())
        self.max_abs_1 = lambda: tfp.bijectors.AffineScalar(
            shift=f64(-2.0), scale=f64(4.0)
        )(tfp.bijectors.Sigmoid())

    def _train(self, x, y, epoch=10, minibatch_size=128, x_test=None, y_test=None):
        # Use TF Dataset for minibatching
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(x.shape[0])
        elbo = []

        for e in range(epoch):
            train_iter = iter(train_dataset.batch(minibatch_size))
            elbo_mov_avg = []
            for (x_batch, y_batch) in train_iter:
                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)
                    obj = -self.model.maximum_log_likelihood_objective(
                        (x_batch, y_batch)
                    )
                    grads = tape.gradient(obj, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )
                elbo_mov_avg.append(obj.numpy())

            elbo.append(np.mean(elbo_mov_avg))

            print(f"Epoch {int(self.ckpt.epoch)} Elbo: {elbo[-1]}")

            nlpd = self.nlpd(x_test, y_test, minibatch_size)
            test_acc = self.test(x_test, y_test, minibatch_size)
            self.log_data(elbo[-1], nlpd, test_acc)

            self.save_checkpoint()

        return elbo

    def train(
        self, x_train, y_train, epoch=50, minibatch_size=128, x_test=None, y_test=None
    ):

        # TODO: Better way?
        if isinstance(epoch, list):
            set_trainable(self.model.inducing_variable, False)
            set_trainable(self.model.kernel.base_kernel.variance, False)
            set_trainable(self.model.kernel.weights, False)

            print("Training only length scales...")
            self._train(
                x_train,
                y_train,
                epoch[0],
                minibatch_size=minibatch_size,
                x_test=x_test,
                y_test=y_test,
            )

            print("Training base kernel's variance too...")
            set_trainable(self.model.kernel.base_kernel.variance, True)
            self._train(
                x_train,
                y_train,
                epoch[1],
                minibatch_size=minibatch_size,
                x_test=x_test,
                y_test=y_test,
            )

            print("Training Inducing Variables too...")
            set_trainable(self.model.inducing_variable, False)
            self._train(
                x_train,
                y_train,
                epoch[2],
                minibatch_size=minibatch_size,
                x_test=x_test,
                y_test=y_test,
            )

            epoch = epoch[3]

        print("Training all parameters...")
        set_trainable(self.model.kernel.weights, True)
        self._train(
            x_train,
            y_train,
            epoch,
            minibatch_size=minibatch_size,
            x_test=x_test,
            y_test=y_test,
        )