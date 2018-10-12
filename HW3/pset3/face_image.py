import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io
import scipy.misc

from pca import feature_normalize, get_usv, project_data, recover_data


def get_datum_img(row):
    """
    Creates an image object from a single np array with shape 1x1024
    :param row: a single np array with shape 1x1024
    :return: the constructed image
    """
    # pass

    return row.reshape([32,32]).T


def display_data(samples, num_rows=10, num_columns=10):
    """
    Function that picks the first 100 rows from X, creates an image from each,
    then stitches them together into a 10x10 grid of images, and shows it.
    """
    width, height = 32, 32
    num_rows, num_columns = num_rows, num_columns

    big_picture = np.zeros((height * num_rows, width * num_columns))

    row, column = 0, 0
    for index in range(num_rows * num_columns):
        if column == num_columns:
            row += 1
            column = 0
        img = get_datum_img(samples[index])
        big_picture[row * height:row * height + img.shape[0], column * width:column * width + img.shape[1]] = img
        column += 1
    plt.figure(figsize=(10, 10))
    img = scipy.misc.toimage(big_picture)
    plt.imshow(img, cmap=pylab.gray())


def main():
    datafile = 'data/faces.mat'
    mat = scipy.io.loadmat(datafile)
    samples = mat['X']
    display_data(samples)
    # Feature normalize
    samples_norm = feature_normalize(samples)

    # Run SVD
    U, S, Vh = get_usv(samples_norm)

    # Visualize the top 36 eigenvectors found
    print('Top 36 principal component is ', U[:, :36])

    # Project each image down to 36 dimensions
    z = project_data(samples_norm, U, 36)

    # Attempt to recover the original data
    recovered_samples = recover_data(z, U, 36)
    # Plot the dimension-reduced data
    display_data(recovered_samples)

     # Project each image down to 100 dimensions
    z_100 = project_data(samples_norm, U, 100)
    # Attempt to recover the original data
    recovered_samples_100 = recover_data(z_100, U, 100)
    # Plot the dimension-reduced data
    display_data(recovered_samples_100)
    plt.show()

if __name__ == '__main__':
    main()
