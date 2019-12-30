import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


def read_transformed_images(transformed_images_path):
    print("Reading the transformed images file located at path " "'{}'...".format(transformed_images_path))
    images = []
    labels = []
    with open(transformed_images_path) as transformed_images_file:
        reader = csv.reader(transformed_images_file, delimiter=',')
        for line in reader:
            if not line:
                continue
            label = line[0]
            labels.append(label)
            image = line[1:]
            image_int = [int(pixel) for pixel in image]
            image = np.array(image_int)
            images.append(image)
    print("Done!\n")
    return images, labels

def generate_svm_classifier():
    print("Generating SVM model...")
    classifier_model = svm.SVC(kernel ='linear', C=10)
    print("Done!\n")
    return classifier_model


def divide_data_train_test(images, labels, ratio):
    print("Dividing dataset in the ratio '{}' using ""train_test_split():".format(ratio))
    """
    https://medium.com/@contactsunny/how-to-split-your-dataset-to-train-and-test-datasets-using-scikit-learn-e7cf6eb5e0d
    """
    ret = train_test_split(images, labels, test_size=ratio, random_state=42)
    print("Done!\n")
    return ret

def print_with_precision(num):
    return "%0.5f" % num


def init_train_svm_classifier():
    model_output_dir_path = ('./dados/labels/result/')
    model_stats_file_path = os.path.join(model_output_dir_path, "stats-svm.txt")
    print("Model stats will be written to the file at path '{}'.".format(model_stats_file_path))

    os.makedirs(os.path.dirname(model_stats_file_path), exist_ok=True)
    print(model_stats_file_path)
    with open(model_stats_file_path, "w") as model_stats_file:
        images, labels = read_transformed_images('./dados/labels/images_transformed.csv')
        classifier_model = generate_svm_classifier()
        model_stats_file.write("Classifier model details:\n{}\n\n".format(classifier_model))

        training_images, testing_images, training_labels, testing_labels = divide_data_train_test(images, labels, 0.3)

        print("Training the model...")
        classifier_model = classifier_model.fit(training_images, training_labels)
        print("Done!\n")


        model_serialized_path = os.path.join(model_output_dir_path, 'model-serialized-svm.pkl')
        print("Dumping the trained model to disk at path '{}'...".format(model_serialized_path))
        joblib.dump(classifier_model, model_serialized_path)
        print("Dumped\n")

        print("Writing model stats to file...")
        score = classifier_model.score(testing_images, testing_labels)
        accuracy = cross_val_score(classifier_model, testing_images, testing_labels, scoring='accuracy', cv = 10).mean() * 100
        print("Accuracy of SVM is: " , accuracy)
        model_stats_file.write("Model score:\n{}\n\n".format(accuracy))

        predicted = classifier_model.predict(testing_images)
        cnf_matrix = metrics.confusion_matrix(testing_labels, predicted)
        norm_conf = []
        for i in cnf_matrix:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = cnf_matrix.shape

        for x in range(width):
            for y in range(height):
                ax.annotate(str(cnf_matrix[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
        plt.xticks(range(width), alphabet[:width])
        plt.yticks(range(height), alphabet[:height])
        plt.savefig('confusion_matrix.png', format='png')
        plt.show()

        print("raw accuracy_score: %0.2f" % accuracy_score(testing_labels, predicted))
        report = metrics.classification_report(testing_labels, predicted)
        model_stats_file.write("Classification report:\n{}\n\n".format(report))
        print("Done!\n")

        print("Finished!\n")
