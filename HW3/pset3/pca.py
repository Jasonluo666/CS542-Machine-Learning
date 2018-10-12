import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io


def feature_normalize(samples):
    """
    Feature-normalize samples
    :param samples: samples.
    :return: normalized feature
    """
    # pass
    
    return (samples - np.mean(samples, axis=0)) / np.std(samples, axis=0)


def get_usv(sample_norm):
    # pass

    covariance_matrix = np.matmul(sample_norm.T, sample_norm) / sample_norm.shape[0]
    return scipy.linalg.svd(covariance_matrix)

def project_data(samples, U, K):
    """
    Computes the reduced data representation when
    projecting only on to the top "K" eigenvectors
    """

    # Reduced U is the first "K" columns in U
    # pass

    Reduced_U = U[:,:K]
    return np.matmul(samples, Reduced_U)


def recover_data(Z, U, K):
    # pass

    Reduced_U = np.matrix(U[:,:K])
    return np.array(np.matmul(Z, Reduced_U.I))


def main():
    datafile = 'data/data1.mat'
    mat = scipy.io.loadmat(datafile)
    samples = mat['X']

    plt.figure(figsize=(7, 7))
    plt.scatter(samples[:, 0], samples[:, 1], s=30, facecolors='none', edgecolors='b')
    plt.title("Example Dataset", fontsize=18)
    plt.grid(True)
    plt.show()

    
    # Feature normalize

    samples_norm = feature_normalize(samples)

    # Run SVD

    U, S, Vh = get_usv(samples_norm)

    # output the top principal component (eigen- vector) found
    # should expect to see an output of about [-0.707 -0.707]"
    print('Top principal component is ', U[:, 0])


    plt.figure(figsize=(7, 7))
    plt.scatter(samples[:, 0], samples[:, 1], s=30, facecolors='none', edgecolors='b')
    plt.title("Example Dataset: PCA Eigenvectors Shown", fontsize=18)
    plt.xlabel('x1', fontsize=18)
    plt.ylabel('x2', fontsize=18)
    plt.grid(True)
    # To draw the principal component, you draw them starting
    # at the mean of the data

    # IMPLEMENT PLOT
    mean_point = np.mean(samples, axis=0)
    first_principal_component = mean_point + U[:, 0]
    second_principal_component = mean_point + U[:, 1]

    plt.plot([mean_point[0], first_principal_component[0]], [mean_point[1], first_principal_component[1]], \
    c='red', label = 'First Principal Component')

    plt.plot([mean_point[0], second_principal_component[0]], [mean_point[1], second_principal_component[1]], \
    c='pink', label = 'Second Principal Component')

    plt.legend(loc=4)
    plt.show()

    # project the first example onto the first dimension
    # should see a value of about 1.481"

    z = project_data(samples_norm, U, 1)
    print('Projection of the first example is %0.3f.' % float(z[0]))
    recovered_sample = recover_data(z, U, 1)
    print('Recovered approximation of the first example is ', recovered_sample[0])
    print(type(samples_norm),type(recovered_sample))
    plt.figure(figsize=(7, 7))
    plt.scatter(samples_norm[:, 0], samples_norm[:, 1], s=30, facecolors='none', edgecolors='b', label='Original Data Points')
    plt.scatter(recovered_sample[:, 0], recovered_sample[:, 1], s=30, facecolors='none', edgecolors='r', label='PCA Reduced Data Points')

    plt.title("Example Dataset: Reduced Dimension Points Shown", fontsize=14)
    plt.xlabel('x1 [Feature Normalized]', fontsize=14)
    plt.ylabel('x2 [Feature Normalized]', fontsize=14)
    plt.grid(True)

    for x in range(samples_norm.shape[0]):
        plt.plot([samples_norm[x, 0], recovered_sample[x, 0]], [samples_norm[x, 1], recovered_sample[x, 1]], 'k--')

    plt.legend(loc=4)
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2.5, 2.5))
    plt.show()


if __name__ == '__main__':
    main()
