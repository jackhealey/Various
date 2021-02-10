#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from scipy import stats
from sklearn.decomposition import PCA

#train_set, train_labels, test_set, test_labels = load_data()

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

train_set, train_labels, test_set, test_labels = load_data()

class_colours = [CLASS_1_C, CLASS_2_C, CLASS_3_C]

col_list = []
for k in range(len(train_labels)):
    n = train_labels[k]
    c = class_colours[int(n-1)]
    col_list.append(c)

def feature_selection(train_set, train_labels, **kwargs):
    # write your code hereand make sure you return the features at the end of 
    # the function
    spec = [6,9]
    
    return spec

def data_spec(train_set, test_set, spec):
    train_set_spec = train_set[:,spec]
    test_set_spec = test_set[:,spec]
    
    return train_set_spec, test_set_spec

spec = feature_selection(train_set, train_labels)
train_set_spec, test_set_spec = data_spec(train_set, test_set, spec)
spec3 = [0,6,9]
train_set_spec3, test_set_spec3 = data_spec(train_set, test_set, spec3)

def knn(train_set_spec, train_labels, test_set_spec, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    class_sep = np.hstack((train_labels.reshape(len(train_labels),1), train_set_spec))
    knn_predictions = np.zeros(len(test_set_spec), dtype = np.int)

    for m in range(len(test_set_spec)):
        dist = []
        small_list = np.zeros(k)
        idx = np.zeros(k, dtype = np.int)
        for j in range(len(train_set_spec)):
            dist.append(np.sqrt((test_set_spec[m,0] - train_set_spec[j,0])**2 + (test_set_spec[m,1] - train_set_spec[j,1])**2))

        dist_og = dist

        for i in range(k):
            small_list[i] = min(dist)
            idx[i] = dist.index(min(dist))
            dist.remove(min(dist))
            
            if k == 2:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 3 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 1 and len(np.unique(((class_sep[idx,:])[:,0]))) == 3:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 4 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 2 and len(np.unique(((class_sep[idx,:])[:,0]))) == 2:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 5 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 2 and len(np.unique(((class_sep[idx,:])[:,0]))) == 3:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 7 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 3 and len(np.unique(((class_sep[idx,:])[:,0]))) == 3:
                pred = ((class_sep[idx,:])[:,0])[0]
            else:
                pred = stats.mode(((class_sep[idx,:])[:,0]))[0]
                
        knn_predictions[m] = pred
    
    return knn_predictions


def alternative_classifier(train_set_spec, train_labels, test_set_spec, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    class_sep = np.hstack((train_labels.reshape(len(train_labels),1), train_set_spec))
    centroids = np.zeros((int(max(train_labels)), 2))
    class_vars = np.zeros((int(max(train_labels)), 2))
    alt_predictions = []
    alt_predictions = np.zeros(len(test_set_spec), dtype = np.int)
    
    for i in range(int(max(train_labels))):
        for j in range(len(class_sep)):
            if class_sep[j,0]==(i+1):
                X = np.mean((class_sep[class_sep[:,0]==i+1])[:,1])
                Y = np.mean((class_sep[class_sep[:,0]==i+1])[:,2])
                varX = np.var((class_sep[class_sep[:,0]==i+1])[:,1])
                varY = np.var((class_sep[class_sep[:,0]==i+1])[:,2])
        centroids[i,:] = (X,Y)
        class_vars[i,:] = (varX,varY)
        
    for i in range(len(test_set_spec)):
        pclass_1 = (len(class_sep[class_sep[:,0]==1]) / len(train_labels))*stats.norm(centroids[0,0], class_vars[0,0]).pdf(test_set_spec[i,0])*stats.norm(centroids[0,1],class_vars[0,1]).pdf(test_set_spec[i,1])
        pclass_2 = (len(class_sep[class_sep[:,0]==2]) / len(train_labels))*stats.norm(centroids[1,0], class_vars[1,0]).pdf(test_set_spec[i,0])*stats.norm(centroids[1,1],class_vars[1,1]).pdf(test_set_spec[i,1])
        pclass_3 = (len(class_sep[class_sep[:,0]==3]) / len(train_labels))*stats.norm(centroids[2,0], class_vars[2,0]).pdf(test_set_spec[i,0])*stats.norm(centroids[2,1],class_vars[2,1]).pdf(test_set_spec[i,1])
        pclass = [pclass_1, pclass_2, pclass_3]
        alt_predictions[i] = (pclass.index(max(pclass))+1)
    
    return alt_predictions


def calculate_accuracy(gt_labels, pred_labels):
    # write your code here (remember to return the accuracy at the end!)
    c = 0
    for i in range(len(pred_labels)):
        if gt_labels[i] == pred_labels[i]:
            c = c+1
    acc = (100*c)/len(pred_labels)        
    return acc


def calculate_confusion_matrix(gt_labels, pred_labels):
    # write your code here (remember to return the confusion matrix at the end!)
    comp = np.hstack((gt_labels.reshape(len(test_labels),1), pred_labels.reshape(len(test_labels),1)))
    class_no = max(max(gt_labels),max(pred_labels))
    CM = np.zeros((class_no, class_no))
    
    for i in range(class_no):
        for j in range(class_no):
            c = 0
            truth_array = (comp[comp[:,0]==i+1])[:,1] == j+1
            for k in range(len(truth_array)):
                if truth_array[k] == True:
                    c = c+1
            CM[i,j] = "%.2f" %(c/len((comp[comp[:,0]==i+1])[:,1]))
    
    return CM


def plot_matrix(matrix, ax=None):
    """
    Displays a given matrix as an image.
    
    Args:
        - matrix: the matrix to be displayed        
        - ax: the matplotlib axis where to overlay the plot. 
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`. 
          If you do not explicitily create a figure, then pass no extra argument.  
          In this case the  current axis (i.e. `plt.gca())` will be used        
    """    
    if ax is None:
        ax = plt.gca()
        
    # write your code here
    im = ax.imshow(matrix, cmap=plt.get_cmap('summer'))
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.linspace(0,matrix.shape[0]-1, matrix.shape[0]))
    ax.set_yticks(np.linspace(0,matrix.shape[0]-1, matrix.shape[0]))
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center")


def knn_three_features(train_set_spec, train_labels, test_set_spec, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    class_sep = np.hstack((train_labels.reshape(len(train_labels),1), train_set_spec))
    predictions = np.zeros(len(test_set_spec), dtype = np.int)

    for m in range(len(test_set_spec)):
        dist = []
        small_list = np.zeros(k)
        idx = np.zeros(k, dtype = np.int)
        for j in range(len(train_set_spec)):
            dist.append(np.sqrt((test_set_spec[m,0] - train_set_spec[j,0])**2 + (test_set_spec[m,1] - train_set_spec[j,1])**2) + (test_set_spec[m,2] - train_set_spec[j,2])**2)

        dist_og = dist

        for i in range(k):
            small_list[i] = min(dist)
            idx[i] = dist.index(min(dist))
            dist.remove(min(dist))
            if k == 2:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 3 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 1 and len(np.unique(((class_sep[idx,:])[:,0]))) == 3:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 4 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 2 and len(np.unique(((class_sep[idx,:])[:,0]))) == 2:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 5 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 2 and len(np.unique(((class_sep[idx,:])[:,0]))) == 3:
                pred = ((class_sep[idx,:])[:,0])[0]
            elif k == 7 and (stats.mode(((class_sep[idx,:])[:,0]))[1]) == 3 and len(np.unique(((class_sep[idx,:])[:,0]))) == 3:
                pred = ((class_sep[idx,:])[:,0])[0]
            else:
                pred = stats.mode(((class_sep[idx,:])[:,0]))[0]
                
        predictions[m] = pred
    
    return predictions


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of 
    # the function

    pca = PCA(n_components=2)
    pca.fit(train_set)
    test_pca = pca.transform(test_set)
    train_pca = pca.transform(train_set)

    knn_predictions_pca = knn(train_set_spec, train_labels, test_pca, k)
    knn_predictions_pca = knn(train_pca, train_labels, test_pca, k)
    
    return knn_predictions_pca

'''"KNN CLASSIFIER"

print()
for i in range(5):
    knn_predictions = knn(train_set_spec, train_labels, test_set_spec, k = i+1)
    percen_acc_knn = calculate_accuracy(test_labels, knn_predictions)
    CM_knn = calculate_confusion_matrix(test_labels, knn_predictions)

    print('accuracy of knn (k = ' + str(i+1) + ') classifier: ' + str(percen_acc_knn))

    fig, ax = plt.subplots()
    plt.title('Knn (k = ' + str(i+1) + ') - Confusion matrix')

    plot_matrix(CM_knn, ax=None)

knn_predictions = knn(train_set_spec, train_labels, test_set_spec, k = 7)
percen_acc_knn = calculate_accuracy(test_labels, knn_predictions)
CM_knn = calculate_confusion_matrix(test_labels, knn_predictions)

print('accuracy of knn (k = ' + str(7) + ') classifier: ' + str(percen_acc_knn))

"Naïve Bayes CLASSIFIER"

fig, ax = plt.subplots()
plt.title('Knn (k = ' + str(7) + ') - Confusion matrix')

plot_matrix(CM_knn, ax=None)


alt_predictions = alternative_classifier(train_set_spec, train_labels, test_set_spec)
percen_acc_alt = calculate_accuracy(test_labels, alt_predictions)
CM_alt = calculate_confusion_matrix(test_labels, alt_predictions)

print()
print('accuracy of Naïve Bayes classifier: ' + str(percen_acc_alt))

fig, ax = plt.subplots()
plt.title('Naïve Bayes classifier - Confusion matrix')

plot_matrix(CM_alt, ax=None)

"KNN_3D CLASSIFIER"

print()
for i in range(5):
    knn3d_predictions = knn_three_features(train_set_spec3, train_labels, test_set_spec3, k = i+1)
    percen_acc_knn3d = calculate_accuracy(test_labels, knn3d_predictions)
    CM_knn3d = calculate_confusion_matrix(test_labels, knn3d_predictions)

    print('accuracy of knn3d (k = ' + str(i+1) + ') classifier: ' + str(percen_acc_knn3d))

    fig, ax = plt.subplots()
    plt.title('Knn3d (k = ' + str(i+1) + ') - Confusion matrix')

    plot_matrix(CM_knn3d, ax=None)

knn3d_predictions = knn_three_features(train_set_spec3, train_labels, test_set_spec3, k = 7)
percen_acc_knn3d = calculate_accuracy(test_labels, knn3d_predictions)
CM_knn3d = calculate_confusion_matrix(test_labels, knn3d_predictions)

print('accuracy of knn3d (k = ' + str(7) + ') classifier: ' + str(percen_acc_knn3d))
print()

fig, ax = plt.subplots()
plt.title('Knn3d (k = ' + str(7) + ') - Confusion matrix')

plot_matrix(CM_knn3d, ax=None)

"PCA CLASSIFIER"

for i in range(5):
    knn_predictions_pca, train_pca = knn_pca(train_set, train_labels, test_set, k=i+1)
    percen_acc_pca = calculate_accuracy(test_labels, knn_predictions_pca)
    CM_pca = calculate_confusion_matrix(test_labels, knn_predictions_pca)
    
    print('accuracy of PCA - knn (k = ' + str(i+1) + ') classifier: ' + str(percen_acc_pca))

    fig, ax = plt.subplots()
    plt.title('PCA (k = ' + str(i+1) + ') - Confusion matrix')

    plot_matrix(CM_pca, ax=None)

knn_predictions_pca, train_pca = knn_pca(train_set, train_labels, test_set, k=7)
percen_acc_pca = calculate_accuracy(test_labels, knn_predictions_pca)
CM_pca = calculate_confusion_matrix(test_labels, knn_predictions_pca)
    
print('accuracy of PCA - knn (k = ' + str(7) + ') classifier: ' + str(percen_acc_pca))
print()

fig, ax = plt.subplots()
plt.title('PCA (k = ' + str(7) + ') - Confusion matrix')

plot_matrix(CM_pca, ax=None)

fig, ax = plt.subplots()
ax.set_title("PCA reduced Training Set")
ax.scatter(train_pca[:,0], -train_pca[:,1] , c = col_list[:])
plt.show()'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')
    
    args = parser.parse_args()
    mode = args.mode[0]
    
    return args, mode


if __name__ == '__main__':
    args, mode = parse_args() # get argument from the command line
    
    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path, 
                                                                       train_labels_path=args.train_labels_path,
                                                                       test_set_path=args.test_set_path,
                                                                       test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set_spec, train_labels, test_set_spec, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set_spec, train_labels, test_set_spec)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(train_set_spec3, train_labels, test_set_spec3, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))

