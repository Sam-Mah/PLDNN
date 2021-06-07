# This code runs parallelly a Neural Network with and without Pseudolabeling

# import tkinter as TK
import matplotlib
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from scipy import stats
from sklearn.preprocessing import Normalizer
from random import shuffle
from sklearn.preprocessing import minmax_scale
import math
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow_confusion_metrics import tf_confusion_metrics
from tensorflow_confusion_metrics import tf_confusion_metrics_2
import random
import pandas as pd
from sklearn import metrics
import datetime
from sklearn.manifold import TSNE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cat = 5


def balance_data(dataset, num_per_label, col):
    # pick out the same size label from data set
    counter = np.zeros(cat)  # for 5 classes
    new_dataset = []
    for i in dataset:
        y = i[col]
        if type(y) == np.ndarray:
            y = np.argmax(y)
        if y == 1 and counter[0] < num_per_label:
            new_dataset.append(i)
            counter[0] += 1
            continue
        if y == 2 and counter[1] < num_per_label:
            new_dataset.append(i)
            counter[1] += 1
            continue
        if y == 3 and counter[2] < num_per_label:
            new_dataset.append(i)
            counter[2] += 1
            continue
        if y == 4 and counter[3] < num_per_label:
            new_dataset.append(i)
            counter[3] += 1
            continue
        if y == 5 and counter[4] < num_per_label:
            new_dataset.append(i)
            counter[4] += 1
            continue
        # if y == 5 and counter[5] < num_per_label:
        #     new_dataset.append(i)
        #     counter[5] += 1
        #     continue
        #
        # if y == 6 and counter[6] < num_per_label:
        #     new_dataset.append(i)
        #     counter[6] += 1
        #     continue
        #
        # if y == 7 and counter[7] < num_per_label:
        #     new_dataset.append(i)
        #     counter[7] += 1
        #     continue
        #
        # if y == 8 and counter[8] < num_per_label:
        #     new_dataset.append(i)
        #     counter[8] += 1
        #
        #     continue
        #
        # if y == 9 and counter[9] < num_per_label:
        #     new_dataset.append(i)
        #     counter[9] += 1
        #     continue

    random.shuffle(new_dataset)
    print(counter)
    new_dataset = np.array(new_dataset)
    print(type(new_dataset))
    return new_dataset


def outlier_detection(data):

    # #z-score outlier detection
    # z = np.abs(stats.zscore(data))
    # print(len(z))
    # clean_data = data[(z < 6).all(axis=1)]
    #
    # print(len(clean_data))

    # EllipticEnvelope from Scikit Learn
    outliers_fraction = 0.35
    clusters_separation = [0, 1, 2]
    n_samples = data.shape[0]

    X = data
    # define two outlier detection tools to be compared
    classifiers = {
        "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
                                         kernel="rbf", gamma=0.1),
        "robust covariance estimator": EllipticEnvelope(contamination=.1)}

    # Compare given classifiers under given settings
    xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    ground_truth = np.ones(n_samples, dtype=int)
    ground_truth[-n_outliers:] = 0

    # Fit the problem with varying cluster separation
    for i, offset in enumerate(clusters_separation):
        np.random.seed(42)
        # # Data generation
        # X1 = 0.3 * np.random.randn(0.5 * n_inliers, 2) - offset
        # X2 = 0.3 * np.random.randn(0.5 * n_inliers, 2) + offset
        # X = np.r_[X1, X2]
        # # Add outliers
        # X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

        # Fit the model with the One-Class SVM
        # plt.figure(figsize=(10, 5))
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            # fit the data and tag outliers
            clf.fit(X)
            y_pred = clf.decision_function(X).ravel()
            threshold = stats.scoreatpercentile(y_pred,
                                                100 * outliers_fraction)
            y_pred = y_pred > threshold
            n_errors = (y_pred != ground_truth).sum()
            # plot the levels lines and the points
        #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #     Z = Z.reshape(xx.shape)
        #     subplot = plt.subplot(1, 2, i + 1)
        #     subplot.set_title("Outlier detection")
        #     subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
        #                      cmap=plt.cm.Blues_r)
        #     a = subplot.contour(xx, yy, Z, levels=[threshold],
        #                         linewidths=2, colors='red')
        #     subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
        #                      colors='orange')
        #     b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
        #     c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')
        #     subplot.axis('tight')
        #     subplot.legend(
        #         [a.collections[0], b, c],
        #         ['learned decision function', 'true inliers', 'true outliers'],
        #         prop=matplotlib.font_manager.FontProperties(size=11))
        #     subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        #     subplot.set_xlim((-7, 7))
        #     subplot.set_ylim((-7, 7))
        # plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)

    plt.show()

    clf.predict(X)
    clean_data = X[clf.predict(X) == 1]

    print("Number of data after cleaning", len(clean_data))

    return clean_data


def normalize(data):

    normalized_data = Normalizer(norm='l2').fit_transform(data)

    return normalized_data




# data = np.genfromtxt(
#     "feature_vectors_syscalls_frequency_5_Cat.csv", delimiter=",")

data = np.genfromtxt(
    "feature_vectors_syscallsbinders_frequency_5_Cat.csv", delimiter=",")

print("Number of data before cleaning", len(data))

# data = outlier_detection(data)

col_num = data.shape[1]


# data = balance_data(data, 1253, col_num - 1)

labels = np.array(data[:, col_num - 1])
labels = labels.astype(int)
print(np.unique(labels))

# dropping the labels' columns
data = data[:, :col_num - 1]

data = normalize(data)

# 1-d array to one-hot conversion
onehot_labels = np.zeros((labels.shape[0], cat))
onehot_labels[np.arange(labels.size), labels - 1] = 1

# print(len(data[np.where(data>=300000)]))
#
# data[data > 300000]= 0
#
# print(data[np.where(data>=300000)])

train_data, test_data, train_labels, test_labels = train_test_split(
    data, onehot_labels, test_size=0.3)

print(type(train_labels))

s = np.array([np.where(r == 1)[0][0] for r in train_labels])
# s = s.astype(int)
print(np.unique(s))
print("Adware_training=", (s == 0).sum())
print("Banking_training=", (s == 1).sum())
print("SMS_training", (s == 2).sum())
print("Riskware_training=", (s == 3).sum())
print("Benign_training", (s == 4).sum())

print(type(test_labels))
s = np.array([np.where(r == 1)[0][0] for r in test_labels])
# s = s.astype(int)
print("Adware_testing=", (s == 0).sum())
print("Banking_testing=", (s == 1).sum())
print("SMS_testing", (s == 2).sum())
print("Riskware_testing=", (s == 3).sum())
print("Benign_testing", (s == 4).sum())

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# Neural Network parameters
iteration_list = np.zeros(1)
neural_network_accuracy_list = np.zeros(1)
pseudo_label_accuracy_list = np.zeros(1)
neural_network_cost_list = np.zeros(1)
pseudo_label_cost_list = np.zeros(1)

neural_network_accuracy = 0
pseudo_label_accuracy = 0


learningRate = 0.4
# learningRate = 0.4
trainingEpochs = 1500
# trainingEpochs = 1500

lbl_samples = 1000

# inputN = 139
# hiddenN = 100 # {130 120 100 80 60 40}
# hiddenN2 = 70 # {100 80 60 40 30 20}
# hiddenN3 = 50 # {60 40 20 15 10 8}
# hiddenN4 = 30 # {40 20 10 8 5}
# hiddenN5 = 15 # {20 10 8 5 3}
# outputN = cat

inputN = 470
hiddenN = 400 # {400 350 300}
hiddenN2 = 250 # {300 250 200}
hiddenN3 = 150 # {200 150 100}
hiddenN4 = 30 # {100 50 30}
hiddenN5 = 10 # {30 20 10}
outputN = cat

batchSize = 100
num_train_samples = train_data.shape[0]
PLbatchSize = math.ceil(
    ((num_train_samples-lbl_samples)*batchSize)/lbl_samples)
# PLbatchSize = 712

iteration = 0
epoch = 0
cPL = 0
T1 = 100
T2 = 400
a = 0.
af = 1.5

print("HiddenLayer1:", hiddenN, "HiddenLayer2:", hiddenN2, "HiddenLayer3:",
      hiddenN3, "HiddenLayer4:", hiddenN4, "HiddenLayer5:", hiddenN5)

x = tf.placeholder("float", [None, inputN])
y = tf.placeholder("float", [None, outputN])
PLx = tf.placeholder("float", [None, inputN])
PLy = tf.placeholder("float", [None, outputN])
alpha = tf.placeholder("float", )

# plt.clf()

def NN(x, w, b):
    # Hidden layer 1
    HL = tf.add(tf.matmul(x, w['h1']), b['b1'])
    HL = tf.nn.sigmoid(HL)  # sigmoid(HL)

    # Hidden layer 2
    HL2 = tf.add(tf.matmul(HL, w['h2']), b['b2'])
    HL2 = tf.nn.sigmoid(HL2)  # sigmoid(HL)

    # Hidden layer 3
    HL3 = tf.add(tf.matmul(HL2, w['h3']), b['b3'])
    HL3 = tf.nn.sigmoid(HL3)  # sigmoid(HL)

    # Hidden layer 4
    HL4 = tf.add(tf.matmul(HL3, w['h4']), b['b4'])
    HL4 = tf.nn.sigmoid(HL4)  # sigmoid(HL)

    # Hidden layer 5
    HL5 = tf.add(tf.matmul(HL4, w['h5']), b['b5'])
    HL5 = tf.nn.sigmoid(HL5)  # sigmoid(HL)

    # Output layer
    out_layer = tf.matmul(HL5, w['out']) + b['out']

    return out_layer


# initialize weights and biases

weightsNN = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),
    'h2': tf.Variable(tf.random_normal([hiddenN, hiddenN2])),
    'h3': tf.Variable(tf.random_normal([hiddenN2, hiddenN3])),
    'h4': tf.Variable(tf.random_normal([hiddenN3, hiddenN4])),
    'h5': tf.Variable(tf.random_normal([hiddenN4, hiddenN5])),
    'out': tf.Variable(tf.random_normal([hiddenN5, outputN]))
}
biasesNN = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),
    'b2': tf.Variable(tf.random_normal([hiddenN2])),
    'b3': tf.Variable(tf.random_normal([hiddenN3])),
    'b4': tf.Variable(tf.random_normal([hiddenN4])),
    'b5': tf.Variable(tf.random_normal([hiddenN5])),
    'out': tf.Variable(tf.random_normal([outputN]))
}
weightsPL = {
    'h1': tf.Variable(tf.random_normal([inputN, hiddenN])),
    'h2': tf.Variable(tf.random_normal([hiddenN, hiddenN2])),
    'h3': tf.Variable(tf.random_normal([hiddenN2, hiddenN3])),
    'h4': tf.Variable(tf.random_normal([hiddenN3, hiddenN4])),
    'h5': tf.Variable(tf.random_normal([hiddenN4, hiddenN5])),
    'out': tf.Variable(tf.random_normal([hiddenN5, outputN]))
}
biasesPL = {
    'b1': tf.Variable(tf.random_normal([hiddenN])),
    'b2': tf.Variable(tf.random_normal([hiddenN2])),
    'b3': tf.Variable(tf.random_normal([hiddenN3])),
    'b4': tf.Variable(tf.random_normal([hiddenN4])),
    'b5': tf.Variable(tf.random_normal([hiddenN5])),
    'out': tf.Variable(tf.random_normal([outputN]))
}
currentDT_1 = datetime.datetime.now()
print (str(currentDT_1))
predNN = NN(x, weightsNN, biasesNN)
predPL = NN(x, weightsPL, biasesPL)
predPL1 = NN(PLx, weightsPL, biasesPL)

costNN = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predNN,
                                                                labels=y))

costPL = tf.add(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predPL,
                                                                       labels=y)),
                (alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predPL1,
                                                                                labels=PLy))))

# Gradient Descent
optimizerNN = tf.train.GradientDescentOptimizer(learningRate).minimize(costNN)
optimizerPL = tf.train.GradientDescentOptimizer(learningRate).minimize(costPL)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
def accuracytestNN(sess):

    return tf_confusion_metrics(predNN, y, sess, {x: test_data, y: test_labels})


def accuracytestPL(sess):



    return tf_confusion_metrics(predPL, y, sess, {x: test_data, y: test_labels})

def roc_curve(sess):

    probs = tf.nn.softmax(predPL)
    predictions = sess.run(probs, {x: test_data, y: test_labels})
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

    plt.plot(tpr, fpr)
    plt.show()

# def tsne(sess):
    # plt.figure(figsize=(10, 5))
    # # colors = 'r', 'b'
    # # target_ids = -1.0, 1.0
    # # target_names ='phishing','legitimate'
    #
    # colors = 'r', 'b' , 'g' , 'c' , ''
    # target_ids = 1, 2 , 3 , 4, 5
    # target_names ='Iris-setosa','Iris-versicolor' , 'Iris-virginica'
    #
    # X_tsne = TSNE(learning_rate=100, perplexity= 1000, n_iter=5000).fit_transform(input_data)
    # for i, c, lbl in zip(target_ids, colors, target_names):
    #     plt.scatter(X_tsne[label==i, 0], X_tsne[label==i, 1], c=c, label=lbl)
    #
    # plt.title('TSNE')
    # plt.legend()
    # plt.show()

def getBatch(list, batchSize):

    try:
        for i in range(0, len(list), batchSize):
            yield list[i:i + batchSize]
    except Exception as E:
        print(E)


with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    # for epoch in range(trainingEpochs):
    avg_costNN = 1.
    avg_costPL = 1.
    while (avg_costNN > 0.05 and epoch < trainingEpochs):

        avg_costNN = 0.
        avg_costPL = 0.

        total_batch = int(lbl_samples / batchSize)

        c = list(zip(train_data, train_labels))
        random.shuffle(c)

        train_data, train_labels = zip(*c)

        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)

        data_batches = list(getBatch(train_data[0:lbl_samples, :], batchSize,))
        labels_batches = list(
            getBatch(train_labels[0:lbl_samples, :], batchSize))

        PLdata_batches = list(
            getBatch(train_data[lbl_samples:, :], PLbatchSize))
        PLlabels_batches = list(
            getBatch(train_labels[lbl_samples:, :], PLbatchSize))

        # Loop over all batches
        try:
            for i in range(total_batch):

                # batch_x, batch_y = mnist.train.next_batch(batchSize)
                batch_x = data_batches[i]
                batch_y = labels_batches[i]

                _, cNN = sess.run([optimizerNN, costNN], feed_dict={x: batch_x,
                                                                    y: batch_y})

                # implementation of alpha calculation formula
                if iteration >= T1:
                    a = ((iteration - T1) / (T2 - T1)) * af
                    if iteration >= T2:
                        a = af

                # Pseudolabel // determine the class or label of each corresponding unlabeled samples
                # batch_xpred, _ = mnist.train.next_batch(PLbatchSize)
                batch_xpred = PLdata_batches[i]
                batch_ypred = sess.run([predPL], feed_dict={x: batch_xpred})
                batch_ypred = batch_ypred[0]
                batch_ypred = batch_ypred.argmax(1)
                # print(batch_ypred)
                batch_ypre = np.zeros((len(batch_ypred), cat))
                # print(i, len(batch_ypred))
                for ii in range(len(batch_ypred)):
                    batch_ypre[ii, batch_ypred[ii]] = 1

                _, cPL = sess.run([optimizerPL, costPL], feed_dict={x: batch_x,
                                                                    y: batch_y,
                                                                    PLx: batch_xpred,
                                                                    PLy: batch_ypre,
                                                                    alpha: a})
                iteration = iteration + 1
                # Compute average loss
                avg_costNN += cNN
                avg_costPL += cPL
        except Exception as E:
            print(E)
        avg_costNN += avg_costNN / total_batch
        avg_costPL += avg_costPL / total_batch

        if iteration % 100 == 0:
            print('t=', iteration)
            neural_network_accuracy = accuracytestNN(sess)
            print('NN acc=', neural_network_accuracy)
            pseudo_label_accuarcy = accuracytestPL(sess)
            print('PL acc=', pseudo_label_accuarcy)

            print("NN cost: ", avg_costNN)
            print("PL cost: ", avg_costPL)

            iteration_list = np.append(iteration_list, iteration)
            # neural_network_accuracy_list = np.append(
            #     neural_network_accuracy_list, neural_network_accuracy[])
            # pseudo_label_accuracy_list = np.append(
            #     pseudo_label_accuracy_list, pseudo_label_accuarcy)

            neural_network_cost_list = np.append(
                neural_network_cost_list, avg_costNN)
            pseudo_label_cost_list = np.append(
                pseudo_label_cost_list, avg_costPL)

        epoch += 1

    x_NN = accuracytestNN(sess)
    x_PL = accuracytestPL(sess)

    conf = tf_confusion_metrics_2(predPL, y, sess, {x: test_data, y: test_labels})

    sum_conf = np.sum(conf, axis=1)

    lst = []
    for i in range(len(sum_conf)):
        lst.append(np.round((conf[i, :] / sum_conf[i]), 2))

    arr = np.array(lst)

    print("Optimization Finished!")

    print("Confusion Matrix:")
    print(conf)
    print(arr)
    print(sum_conf)
    print("NN:", x_NN)
    print("+PL:", x_PL)

    pd.concat([x_NN, x_PL], axis=1).to_csv('test_5_layers.csv')

    #########################ROC Curve############################

    currentDT_2 = datetime.datetime.now()
    print(str(currentDT_2))

    dateTimeDifference = currentDT_2 - currentDT_1

    print('Run Time in Seconds: ',dateTimeDifference.total_seconds())

    plt.plot(iteration_list, pseudo_label_cost_list, 'r--', label='PLDNN')
    plt.plot(iteration_list, neural_network_cost_list, 'b--', label='DNN')

    plt.legend(loc='upper left')
    plt.xlabel("Iterations")
    plt.ylabel("Average Training Cost")
    plt.ylim(0,26)
    plt.xlim(0,16000)
    plt.savefig('Fig_avg_cost_PLDNN.eps', format='eps')

    # roc_curve(sess)
