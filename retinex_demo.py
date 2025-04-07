import numpy as np
import cv2
from skimage import img_as_float64
import matplotlib.pyplot as plt
from PIL import Image

SIGMA_LIST = [15, 80, 250]
ALPHA = 125.0
BETA = 46.0
G = 5.0
OFFSET = 25.0


def display_image_in_actual_size(img):
    # dpi = 80
    
    # height, width, depth = img.shape
    
    # # What size does the figure need to be in inches to fit the image?
    # figsize = width / float(dpi), height / float(dpi)
    
    # # Create a figure of the right size with one axes that takes up the full figure
    # fig = plt.figure(figsize=figsize)
    # ax = fig.add_axes([0, 0, 1, 1])
    # #
    # # # Hide spines, ticks, etc.
    # ax.axis('off')

    # # Display the image.
    # ax.imshow(img, cmap='gray')
    cv2.imshow('output', img)
    # plt.show()


def singleScale(img, sigma):
    # L = S * G
    L = cv2.GaussianBlur(img, (0, 0), sigma)

    # L = cv2.GaussianBlur(img,(3,3),0)

    # SSR = np.log10(img) - np.log10(cv2.GaussianBlur(img,(0,0),sigma))
    # img = img + 1
    SSR = np.log10(img) - np.log10(L)

    return SSR, L


def multiScale(img, sigmas: list):
    # \omega is 1/3 and equal for all

    retinex = np.zeros_like(img)
    # luminance = np.zeros_like(img)
    for s in sigmas:
        SSR, L = singleScale(img, s)
        retinex += SSR
        # luminance += L

    MSR = retinex / len(sigmas)
    # luminance = luminance/len(sigmas)
    return MSR


def crf(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    i_prime = img / img_sum

    # color_restoration = beta * (np.log10(alpha*img) - np.log10(img_sum))

    color_restoration = beta * (np.log10(alpha * i_prime))

    return color_restoration


def MSRCR(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0, G=5.0, b=25.0):
    """
    MSRCR (Multi-scale retinex with color restoration)

    Parameters :

    img : input image
    sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
    alpha : controls the strength of the nonlinearity
    beta : gain constant
    G : final gain
    b : offset
    """
    Dynamic = 2

    # Mean = np.mean(MSR)
    # Var = np.var(MSR)

    img = img_as_float64(img) + 1

    MSR = multiScale(img, sigmas)
    # C = crf(img, alpha, beta)
    # MSRCR = G * (MSR*C + b)
    MSRCR = np.zeros_like(img)
    for i in range(MSR.shape[2]):
        MSRCR[:, :, i] = 255 * ((MSR[:, :, i] - np.mean(MSR[:, :, i]) + Dynamic * np.var(MSR[:, :, i])) / (
                    2 * Dynamic * np.var(MSR[:, :, i])))

    MSRCR = np.uint8(np.minimum(np.maximum(MSRCR, 0), 255))

    img = img_as_float64(img) + 1

    MSR = multiScale(img, sigmas)
    img_color = crf(img, alpha, beta)
    MSRCR = G * (MSR * img_color + b)

    for i in range(MSRCR.shape[2]):
        MSRCR[:, :, i] = (MSRCR[:, :, i] - np.min(MSRCR[:, :, i])) / (
                    np.max(MSRCR[:, :, i]) - np.min(MSRCR[:, :, i])) * 255

    MSRCR = np.uint8(np.minimum(np.maximum(MSRCR, 0), 255))

    return MSRCR


def bilateral_filter(img, filter_radius=15, sigma_d=5, sigma_r=0.03):
    """
    filter_radius (int): Diameter of each pixel neighborhood.
    sigma_d (float): Filter sigma in the coordinate space (spatial sigma).
    sigma_r (float): Filter sigma in the color space (range sigma).

    """
    sigma_color = sigma_r * 255  # Convert to the range of [0, 255]
    sigma_space = sigma_d
    return cv2.bilateralFilter(img, filter_radius, sigma_color, sigma_space)


def create_gaussian_pyramid(MSRCR_img):
    gaussian = []

    gaussian_layer = MSRCR_img.copy()

    gaussian.append(gaussian_layer)
    # display_image_in_actual_size(gaussian_layer)
    for i in range(3):
        gaussian_layer = cv2.pyrDown(gaussian_layer)
        gaussian.append(gaussian_layer)
        # display_image_in_actual_size(gaussian_layer)
    return gaussian


def create_laplacian_pyramid(MSRCR_img):
    gaussian = create_gaussian_pyramid(MSRCR_img)
    laplacian = [gaussian[-1]]
    for i in range(3, 0, -1):
        size = (gaussian[i - 1].shape[1], gaussian[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian[i], dstsize=size)
        laplacian_layer = cv2.subtract(gaussian[i - 1], gaussian_expanded)
        laplacian.append(laplacian_layer)
        # display_image_in_actual_size(laplacian_layer)
    return laplacian


def gamma_correction(img_L):
    img_L = img_as_float64(img_L)
    sum = img_L.shape[0] * img_L.shape[1] * img_L.shape[2]
    N = 127

    sum_of_values = np.sum(img_L)

    gamma = sum_of_values / (sum * N)

    c = 1

    Ga = c * np.power(img_L, gamma)

    # Normalize image to range [0, 1]
    Ga = (Ga - np.min(Ga)) / (np.max(Ga) - np.min(Ga))

    # back to uint8
    Ga = (Ga * 255).astype(np.uint8)

    return Ga


def retinex_based_laplacian_pyramid_defogging(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    MSRCR_img = MSRCR(image, SIGMA_LIST, ALPHA, BETA, G, OFFSET)

    image = img_as_float64(image)

    img_msr, img_L = singleScale(image, 15)

    bf = bilateral_filter(MSRCR_img)

    laplacian_pyramid = create_laplacian_pyramid(MSRCR_img)

    Lambda = 1.2

    # tmp =Lambda*img_as_float64(laplacian[-1])

    LL = img_as_float64(bf) + Lambda * img_as_float64(laplacian_pyramid[-1])

    Ga = gamma_correction(img_L)

    S = img_as_float64(Ga) * img_as_float64(LL)

    alpha = 0.7

    result = cv2.addWeighted(img_as_float64(LL), alpha, img_as_float64(S), 1 - alpha, 0)

    return result


# image_path = 'E:\ImageDefogging\src hugging face\DehazeFormer_Demo\examples\Screenshot (2).png'
image_path = r'/content/fog_road.png'
# display_image_in_actual_size(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
# display_image_in_actual_size(retinex_based_laplacian_pyramid_defogging(image_path))

# cv2.imshow('output' , retinex_based_laplacian_pyramid_defogging(image_path))
# output = Image.fromarray(output.astype(np.uint8))
# output.show()

output = retinex_based_laplacian_pyramid_defogging(image_path)
# output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
# cv2.imshow('output', output)
from google.colab.patches import cv2_imshow  # Import the function
cv2_imshow(output) 

result = cv2.normalize(output, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('output.png', result)

# cv2.imwrite('output.png', output)
cv2.waitKey(0)
