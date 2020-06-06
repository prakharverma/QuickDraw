from ConvGP import ConvGP

import sys
sys.path.append("..")
import utils


if __name__ == "__main__":

    npy_file_path = r"../../sample/output"
    train_n = 500
    test_n = 10
    minibatch_size = 4
    epochs = 10
    n_classes = 2
    lr = 0.005
    n_patches = 500
    patch_shape = (5, 5)
    model_output_path = "../../model/conv_gp/"

    x_train, y_train, x_test, y_test = utils.load_dataset(
        npy_file_path=npy_file_path,
        train_n=train_n,
        test_n=test_n
    )

    model = ConvGP(x_train, y_train, n_patches=n_patches, n_classes=n_classes, patch_shape=patch_shape, lr=lr,
                   output_model_path=model_output_path)

    if model:

        model.train(
            x_train, y_train, epochs, minibatch_size, x_test, y_test
        )

        acc = model.test(x_test, y_test, minibatch_size)
        print(f"Final Mean Accuracy : {acc}")

        nlpd = model.nlpd(x_test, y_test, minibatch_size)
        print(f"Mean NLPD value: {nlpd}")
