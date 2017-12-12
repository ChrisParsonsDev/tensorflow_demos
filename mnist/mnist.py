import matplotlib.pyplot as plt
import numpy as np
import random as rand
import tensorflow as tf
from datetime import datetime

# Import MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# set size of training / testing data
# Defaults: train = 55000 test = 10000
train_size = 55000
test_size = 10000

# Define size of train/val data
# default = train
def split_data(num, Mode="train"):

    if(Mode == "test"):
        X_test = mnist.test.images[:num,:]
        y_test = mnist.test.labels[:num,:]
        print("Total test images loaded: " + str(X_test.shape[0]))
        return X_test, y_test

    if(Mode == "train"):
        X_train = mnist.train.images[:num,:]
        y_train = mnist.train.labels[:num,:]
        print("Total training images loaded: " + str(X_train.shape[0]))
        return X_train, y_train

    return "Error: please specify train/test mode"

# Measure total runtime for TF.
executionStart = datetime.now()

# Create TF Session
sess = tf.Session()

# Placeholder to feed training data into
X_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

# Define weights and biases - initialised to 0
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# init Classifier
y = tf.nn.softmax(tf.matmul(X_,W) + b)

# Specify number of train samples to load Mode="train"
X_train, y_train = split_data(train_size, Mode="train")
# Specify number of test samples to load Mode="test"
X_test, y_test = split_data(test_size, Mode="test")

# Default learning_rate = 0.1
# Default number of training iterations = 300
learning_rate = 0.1
training_iter = 300

# Initialise TensorFlow variables
init = tf.global_variables_initializer()
sess.run(init)

# Determine loss of model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Define Graident Descent classifier
clf = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
# See if classification matches the label for that image
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# Accuracy of the model = average number of times prediction is correct
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# For every training iteration
for iteration in range(training_iter+1):
    # Take the classifier, and feed it the input data
    sess.run(clf, feed_dict={X_: X_train, y_: y_train})
    # Every 50 runs, determine accuracy by feeding model the test data
    if iteration % 50 == 0:
        print("Training Step: "+ str(iteration) + "  Accuracy =  " + str(sess.run(accuracy, feed_dict={X_: X_test, y_: y_test})) + "  Loss = " + str(sess.run(cross_entropy, {X_: X_train, y_: y_train})))


# Stop measuring and report runtime
executionEnd = datetime.now() - executionStart
print("Total Training Time: "+str(executionEnd.total_seconds())+"s")

# After the training has completed, visualise heat map of the weights
for digit in range(10):
    plt.subplot(2, 5, digit+1)
    weight = sess.run(W)[:,digit]
    plt.title(digit)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

plt.show()


# Predict an unknown number
def show_prediction(num):
    # This will load one random training sample
    X_train = mnist.validation.images[num,:].reshape(1,784)
    y_train = mnist.validation.labels[num,:]
    # Get the label for the selected image
    label = y_train.argmax()
    # Get prediction for the image
    prediction = sess.run(y, feed_dict={X_: X_train}).argmax()
    # Plot prediction vs label and show image
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(X_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

# How many digits to test
test_runs = 5

for test in range(test_runs):
    # Select random number from the dataset and visualise the prediction
    show_prediction(rand.randint(0,5000))
