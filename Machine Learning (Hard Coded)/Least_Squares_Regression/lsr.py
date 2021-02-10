import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

file_in = sys.argv[1:] # sys.argv contains the arguments passed to the program
filename = str(file_in[0])
points1, points2 = load_points_from_file(filename)

x = points1.reshape((len(points1),1))
y = points2.reshape((len(points2),1))

n = np.array(np.split(x,20)).shape[1]   # n: number of line segments

yplot = []
index_list = []
summat_list = np.zeros((n, 3))
err_min = np.zeros((n,1))

for i in range(n):

    X = np.concatenate((np.ones((20,1)), x[i*20:(i+1)*20]), axis = 1)
    X_o3 = np.concatenate((np.ones((20,1)), x[i*20:(i+1)*20], x[i*20:(i+1)*20]**2, x[i*20:(i+1)*20]**3), axis = 1)
    X_wave = np.concatenate((np.ones((20,1)), np.sin(x[i*20:(i+1)*20])), axis = 1)

    A = (np.linalg.inv((X.T.dot(X)))).dot(X.T).dot(y[i*20:(i+1)*20])
    A_o3 = (np.linalg.inv((X_o3.T.dot(X_o3)))).dot(X_o3.T).dot(y[i*20:(i+1)*20])
    A_wave = (np.linalg.inv((X_wave.T.dot(X_wave)))).dot(X_wave.T).dot(y[i*20:(i+1)*20])

    y_dash = A[0] + A[1]*x[i*20:(i+1)*20]
    y_dash_o3 = A_o3[0] + A_o3[1]*x[i*20:(i+1)*20] + A_o3[2]*x[i*20:(i+1)*20]**2 + A_o3[3]*x[i*20:(i+1)*20]**3
    y_dash_wave = A_wave[0] + A_wave[1]*np.sin(x[i*20:(i+1)*20])

    err = (y[i*20:(i+1)*20] - (A[0] + A[1]*x[i*20:(i+1)*20]))**2
    err_o3 = (y[i*20:(i+1)*20] - (A_o3[0] + A_o3[1]*x[i*20:(i+1)*20] + A_o3[2]*x[i*20:(i+1)*20]**2 + A_o3[3]*x[i*20:(i+1)*20]**3))**2
    err_wave = (y[i*20:(i+1)*20] - (A_wave[0] + A_wave[1]*np.sin(x[i*20:(i+1)*20])))**2

    summat_err = sum(err)
    summat_err_o3 = sum(err_o3)
    summat_err_wave = sum(err_wave)

    summat_list_seg = [summat_err, summat_err_o3, summat_err_wave]

    index = summat_list_seg.index(min(summat_list_seg))
    if index == 1 and abs(A_o3[3]) < 0.01:
        index = 0
        summat_list_seg[1] = summat_err

    index_list.append(index)

    summat_list[i,:] = summat_list_seg
    err_min[i] = min(summat_list[i,:])

    yplot.append(y_dash), yplot.append(y_dash_o3), yplot.append(y_dash_wave)
    
print(float(sum(err_min)), '\n')

if len(sys.argv) == 3 and sys.argv[2] == "--plot":

        fig, ax = plt.subplots()
        plt.title("Reconstructing unknown signal")
        ax.set_xlabel('x co-ordinates')
        ax.set_ylabel('y co-ordinates')

        for i in range(n):
            plt.plot(x[20*i:(i+1)*20],yplot[3*i+index_list[i]], c='r')
        view_data_segments(points1,points2)
    

    
    

    

    

    
