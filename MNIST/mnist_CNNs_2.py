from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

def cnn_model_fn(features, labels, mode):
    """Build a CNN model trained on MNIST data."""

    # Input Layer, 28 x 28 pixel intensities
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer 1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
    conv1 = tf.layers.conv2d(
        input_layer, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu
    )

    # Pooling Layer 1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=[2, 2])

    # Convolutional Layer 2: Applies 64 5x5 filters, with ReLU activation function
    conv2 = tf.layers.conv2d(
        pool1, filters=64, kernel_size=[5, 5], padding='same', activation=tf.nn.relu
    )

    # Pooling Layer 2: Again, performs max pooling with a 2x2 filter and stride of 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=[2, 2])

    # Dense Layer 1: 1,024 neurons, with dropout regularization rate of 0.4 (probability of 0.4 that any given element
    # will be dropped during training)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(dense, rate=0.4, training=(mode == learn.ModeKeys.TRAIN))

    # Dense Layer 2 (Softmax Layer): 10 neurons, one for each digit target class (0-9).
    output = tf.layers.dense(dropout, units=10)

    loss, train_op = None, None

    # Calculating loss for train and eval modes.
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels, output)

    # Configure the training Op, for train mode.
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss, global_step=tf.contrib.framework.get_global_step(), learning_rate=0.001, optimizer='SGD'
        )

    predictions = {
        'classes': tf.argmax(input=output, axis=1),
        'probabilities': tf.nn.softmax(output, name='softmax_tensor')
    }

    # Return the model.
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):

    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model.
    mnist_classifier.fit(x=train_data, y=train_labels, batch_size=100, steps=20000, monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")
    }

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

