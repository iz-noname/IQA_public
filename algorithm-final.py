import cv2
import sys
import math as m
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma as tgamma
from joblib import load


def get_dct_coefficients(img_in_greyscale):
    img_greyscale_float_conversion = np.float32(img_in_greyscale)  # / 255.0
    dct_dst = cv2.dct(img_greyscale_float_conversion)
    return dct_dst


def detect_blur_with_fft(image, size=60, thresh=30, vis=False):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs(fft_shift))
        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    fft_shift = np.fft.ifftshift(fft_shift)
    recon = np.fft.ifft2(fft_shift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


# AGGD fit model, takes input as the MSCN Image / Pair-wise Product
def AGGDfit(structdis):
    # variables to count positive pixels / negative pixels and their squared sum

    poscount = len(structdis[structdis > 0])  # number of positive pixels
    negcount = len(structdis[structdis < 0])  # number of negative pixels

    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum / negcount))
    rsigma_best = np.sqrt((possqsum / poscount))

    gammahat = lsigma_best / rsigma_best

    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum / totalcount, 2) / ((negsqsum + possqsum) / totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1) / (m.pow(m.pow(gammahat, 2) + 1, 2))

    prevgamma = 0
    prevdiff = 1e10
    sampling = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes=[np.float], cache=False)

    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best]


def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while (gam < 10):
        r_gam = tgamma(2 / gam) * tgamma(2 / gam) / (tgamma(1 / gam) * tgamma(3 / gam))
        diff = abs(r_gam - rhatnorm)
        if (diff > prevdiff): break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best


def calculate_BRISQUE_features(img):
    scalenum = 2
    feat = []       # feature vector
    # make a copy of the image
    im_original = img.copy()
    # scale the images twice
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im * im, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(sigma - mu))

        # structdis is the MSCN image
        structdis = im - mu
        structdis /= (sigma + 1.0 / 255)

        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)
        # unwrap the best fit parameters
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best = best_fit_params[2]

        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best * lsigma_best + rsigma_best * rsigma_best) / 2)

        # shifting indices for creating pair-wise products
        shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]  # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift - 1]  # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))

            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product
            # best fit the pairwise product
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best = best_fit_params[2]

            constant = m.pow(tgamma(1 / gamma_best), 0.5) / m.pow(tgamma(3 / gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2 / gamma_best) / tgamma(1 / gamma_best)) * constant

            # append the best fit calculated parameters
            feat.append(gamma_best)  # gamma best
            feat.append(meanparam)  # mean shape
            feat.append(m.pow(lsigma_best, 2))  # left variance square
            feat.append(m.pow(rsigma_best, 2))  # right variance square

        # resize the image on next iteration
        im_original = cv2.resize(im_original, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat


def calculate_entropy(img_greyscale):
    histg = cv2.calcHist([img_greyscale], [0], None, [256], [0, 256])
    h, w = img_greyscale.shape[0], img_greyscale.shape[1]
    histg_1 = []

    for l in histg:
        if (l / (h * w)) > 0:
            histg_1.append((l / (h * w)) * m.log(l / (h * w), 2))
        else:
            histg_1.append(l / (h * w))

    e_temp = -sum(histg_1)
    e_p = (1 / (m.sqrt(2 * m.pi) * 0.8732)) * m.exp((-(e_temp - 7.4600) ** 2) / (2 * (0.8732 ** 2)))
    return e_p


def calculate_feature_vector_for_image(img_greyscale):
    feature_vector = calculate_BRISQUE_features(img_greyscale)
    feature_vector.append(calculate_entropy(img_greyscale))
    feature_vector.append(detect_blur_with_fft(img_greyscale)[0])
    return feature_vector


def measure_quality(img_path):
    global all_features_to_csv
    # read given image (colours)
    img = cv2.imread(img_path)
    # read given image in greyscale
    img_greyscale = cv2.imread(img_path, 0)

    if img is None or img_greyscale is None:
        sys.exit("Image could not be read.")

    features = calculate_feature_vector_for_image(img_greyscale)

    # rescaling the features vector to <-1, 1>

    x = []

    # pre-loaded lists of minimal and maximal values of particular features

    min_ = [0.44, 0.05392, 0.284, -0.077, 0.00182, 0.00474, 0.285, -0.04294, 0.00175, 0.00426, 0.288, -0.09816, 0.0024,
     0.00329, 0.288, -0.11600999999999999, 0.0024100000000000002, 0.0033299999999999996, 0.457, 0.08972000000000001,
     0.298, -0.11448, 0.00509, 0.00999, 0.299, -0.1198, 0.00464, 0.00807, 0.302, -0.09109, 0.006490000000000001,
     0.00881, 0.301, -0.09598, 0.00645, 0.00845, 0.0, -27.47601]

    max_ = [2.921, 0.53273, 0.938, 0.14193, 0.31134, 0.24481, 0.941, 0.12571, 0.28884, 0.24724000000000002, 0.932,
     0.06717000000000001, 0.34209, 0.2056, 0.929, 0.06783, 0.3422, 0.20572, 2.924, 0.51122, 0.935, 0.11316, 0.31431,
     0.24896, 0.931, 0.10818, 0.32807, 0.21261999999999998, 0.94, 0.05244, 0.28726999999999997, 0.2076, 0.941,
     0.05142000000000001, 0.28475, 0.2094, 0.45687, 45.15518]

    for i in range(38):
        min = min_[i]
        max = max_[i]

        # <0, 1>
        x.append((features[i] - min) / (max - min))

    features_asarray = np.asarray(x)
    reshaped_features = features_asarray.reshape(1, -1)
    clf = load('model.joblib')

    return clf.predict(reshaped_features)


# Force user to enter image path
if len(sys.argv) != 2:
    sys.exit("Please give input argument of the image path.")

# Calculate image quality score
quality_score = measure_quality(sys.argv[1])
print("Quality score:", quality_score)
