from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis = 0)
    return x

def get_covariance(dataset):
    return np.dot(dataset.T, dataset) / (len(dataset) - 1)

def get_eig(S, k):
    Lambda, U = eigh(S, subset_by_index = [len(S)-k, len(S)-1])
    Lambda = np.diag(Lambda[::-1])
    U = U[:, ::-1]
    return Lambda, U

def get_eig_prop(S, prop):
    Lambda, U = eigh(S)
    Lambda = Lambda[::-1]
    U = U[:, ::-1]

    total = np.sum(Lambda)
    mask = Lambda / total > prop
    Lambda = np.diag(Lambda[mask])
    U = U[:, mask]
    return Lambda, U


def project_and_reconstruct_image(image, U):
    # image(3000,) U(3000,50) U'(50,3000)
    projected = np.dot(np.transpose(U), image)  # (50,3000) & (3000,) = (50,)
    reconstructed = np.dot(U, projected)   # (3000, 50) & (50,)
    return reconstructed

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    im_orig_fullres = im_orig_fullres.reshape(218, 178, 3)
    im_orig = im_orig.reshape(60,50)
    im_reconstructed = im_reconstructed.reshape(60,50)

    ax1.set_title('Original High Res')
    ax2.set_title('Original')
    ax3.set_title('Reconstructed')

    orig_highres = ax1.imshow(im_orig_fullres, aspect="equal")
    orig = ax2.imshow(im_orig, aspect="equal", cmap = "gray")
    reconstr = ax3.imshow(im_reconstructed, aspect="equal", cmap = "gray")

    fig.colorbar(orig, ax=ax2)
    fig.colorbar(reconstr, ax=ax3)

    plt.show()

    return fig, ax1, ax2, ax3

def perturb_image(image, U, sigma):
    # Your implementation goes here!
    raise NotImplementedError

dataset = load_and_center_dataset("celeba_60x50.npy")
S = get_covariance(dataset)
Lambda, U = get_eig(S, 50)
celeb_idx = 34
x = dataset[celeb_idx]
x_fullres = np.load('celeba_218x178x3.npy')[celeb_idx]
reconstructed = project_and_reconstruct_image(x, U)
fig, ax1, ax2 = display_image(x_fullres, x, reconstructed)