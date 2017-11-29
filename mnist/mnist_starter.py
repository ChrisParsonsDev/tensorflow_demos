import matplotlib.pyplot as plt
import numpy as np
import random as rand
# import TensorFlow

# Load mnist dataset and create mnist variable


# Create two variables called train_size and test_size
# Initialise them to two integers
# Defaults: train = 55000 test = 10000


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

# Create TF Session

# Placeholder to feed training data into


# Define weights and biases - initialised to 0


# init Classifier


# Specify number of train samples to load Mode="train"

# Specify number of test samples to load Mode="test"


# Default learning_rate = 0.1
# Default number of training iterations = 300


# Initialise TensorFlow variables


# Determine loss of model


# Define Graident Descent classifier

# See if classification matches the label for that image

# Accuracy of the model = average number of times prediction is correct

# For every training iteration
for iteration in range(training_iter+1):
    # Take the classifier, and feed it the input data
    sess.run(clf, feed_dict={X_: X_train, y_: y_train})
    # Every 50 runs, determine accuracy by feeding model the test data
    if iteration % 50 == 0:
        print("Training Step: "+ str(iteration) + "  Accuracy =  " + str(sess.run(accuracy, feed_dict={X_: X_test, y_: y_test})) + "  Loss = " + str(sess.run(cross_entropy, {X_: X_train, y_: y_train})))

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
