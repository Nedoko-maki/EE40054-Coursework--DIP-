import matplotlib.pyplot as plt
import numpy as np
import math, scipy, scipy.ndimage
from tqdm import tqdm


def image_size(image_data: np.ndarray):
    print(f"Image is of shape {image_data.shape}, size {image_data.size}, max value {np.max(image_data)}")
    return image_data.shape


def bit_plane_slicing(image_data):
    # find the maximum value, and get the rounded up log2 value 
    # to find the required num of bits needed to represent the array. 
    num_of_bits = math.ceil(math.log(np.max(image_data), 2))  
    
    bit_planes = [np.zeros(image_data.shape, dtype=np.uint8) for _ in range(num_of_bits)]

    for i in range(num_of_bits):
        bit_val = 2 ** (num_of_bits - 1 - i) # input a mask of 00000001 to 10000000 (MSB to LSB)
        bit_planes[i] = (image_data & bit_val)  >> (num_of_bits - 1 - i)  # bitmask the image and shift the image array to one or zeroes.    
 
    return bit_planes


# Some theory: output pixel = f(pixel_array_in_window) 
#
# g(x, y) = h(x, y) <conv> f(x, y)
# G(x, y) = H(u, v) * F(u, v)
#
# capitalised ver. is the Fourier transform of the functions. 
# g = output image
# h = input image


def mean_filter(image_array, window_size=(3, 3), mode="reflect"):
    # if mode = 'reflect', it will mirror the outer values for padding, otherwise mode = 'constant' gives us a chance to pad with a constant. 
    # weights = np.ones(window_size)
    if len(window_size) == 1:
        window_size = (window_size, window_size)

    weights = np.full(window_size, 1.0/(math.prod(window_size)))
    return scipy.ndimage.convolve(image_array, weights, mode=mode)


def _trimmed_mean(window, window_size, exclude_n):
    window_len = np.prod(window_size)
    window = sorted(window)[exclude_n:window_len - exclude_n]
    return np.mean(window)


def trimmed_mean_filter(image_array, window_size=(3, 3), exclude_n=1, mode="reflect"):
    if np.prod(window_size) <= 2 * exclude_n:
        raise ValueError("Can't have that large of an exclude_n, array size after is less than 1.")

    return scipy.ndimage.generic_filter(image_array, 
                                        _trimmed_mean, 
                                        size=window_size, 
                                        extra_arguments=(window_size, exclude_n,), 
                                        mode=mode)


def median_filter(image_array, window_size=3, mode="reflect"):  # Mainly used to remove impulse noise in grey-level images!
    return scipy.ndimage.median_filter(image_array, size=window_size, mode=mode)


def SNR(window):
    mean = np.mean(window)
    std = np.std(window)

    return mean / std


def _weighted_median(window, weights, window_size):
    window = np.multiply(window.reshape(window_size), weights)
    return np.median(window)


def weighted_median_filter(image_array, weights, window_size=(3, 3), mode="reflect"):
    return scipy.ndimage.generic_filter(image_array, 
                                        _weighted_median, 
                                        size=window_size, 
                                        extra_arguments=(weights, window_size), 
                                        mode=mode)


def _adaptive_median(window, window_size, shift, centre_weight, constant):
    positions = np.argwhere(np.full(window_size, 1))  # make a 1-filled matrix with the size of the window, 
    # and get the positions of the elements. 
    distance = np.array([((x[0]-shift)**2 + (x[1]-shift)**2)**(0.5) for x in positions]).reshape(window_size)
    # shift the positions so that 0, 0 is in the centre of the window, and get the distances from 0, 0. 

    epsilon = 1e-13  # avoid zero-divs

    k = (constant * np.std(window)) / (np.mean(window) + epsilon)  # get a constant to reduce the matrix operations needed
    # this doesn't check for zero mean division

    weights = np.full(window_size, centre_weight)
    weights = np.subtract(weights, np.multiply(np.full(window_size, k), distance))

    return _weighted_median(window.reshape(window_size), weights, window_size)  
    # pass off the final calculation to the weighted median f. 


def adaptive_median_filter(image_array, centre_weight, constant, window_size=(3, 3), mode="reflect"):
    shift = window_size[0] // 2  # find the necessary shift for the centre of the window, floored div 2. 
    return scipy.ndimage.generic_filter(image_array, 
                                        _adaptive_median, 
                                        window_size, 
                                        extra_arguments=(window_size, shift, centre_weight, constant), 
                                        mode=mode)


def _geometric_mean(window, k):
    return np.power(np.prod(window), k) 


def geometric_mean_filter(image_array, window_size=(3, 3), mode="reflect"):  
    # most commonly used to filter out Gaussian noise
    k = 1.0/np.prod(window_size)  # save a bit of processing time by preprocessing the weight.
    _filter = scipy.ndimage.generic_filter(image_array, 
                                          _geometric_mean, 
                                          window_size, 
                                          extra_arguments=(k,),
                                          mode=mode)
    return _filter


def _arithmetic_mean(window, window_prod):
    return np.sum(window) / (window_prod)


def arithmetic_mean_filter(image_array, window_size=(3, 3), mode="reflect"):
    _window_prod = np.prod(window_size)  
    # save a bit of processing time by preprocessing the product of the window dims.
    _filter = scipy.ndimage.generic_filter(image_array, 
                                          _arithmetic_mean, 
                                          window_size, 
                                          extra_arguments=(_window_prod,), 
                                          mode=mode)
    return _filter


def histogram_representation(image_array):
    hist, _ = np.histogram(image_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    new_image_array = cdf[image_array]

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(new_image_array.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()


def histogram_equalisation(image_array, h_range=255):
    hist, _ = np.histogram(image_array.flatten(), 256, [0, 256])  
    # generate a histogram
    cdf = hist.cumsum() # get the cumulative sum of the histogram
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * h_range / (cdf_m.max() - cdf_m.min()) 
    # scale the values depending on the range of initial values
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    new_image_array = cdf[image_array]

    return new_image_array


def gaussian_filter(image_array, sigma=1, mode="reflect"):
    return scipy.ndimage.gaussian_filter(image_array, sigma, mode=mode)


def threshold_low_filter(image_array, minVal):
    image_array[image_array <= minVal] = 0 
    return image_array


def _nlm(window, small_window, big_window, Nw):

    half_width = big_window//2
    window = window.reshape((big_window, big_window, small_window, small_window))
    local_window = window[half_width, half_width]
    Ip = evaluate_norm_nlm(local_window, window, Nw)
    
    return max(min(255, Ip), 0) # Clipping the pixel values to stay between 0-255 


def non_local_means_filter(image_array, h, small_window, big_window, mode="reflect"):

    # THIS ISN'T USED BECAUSE IT MEMORY LEAKS OR TAKES WAY TOO 
    # MUCH MEMORY AND I DECIDED THAT USING A NAIVE SOLUTION WOULD SAVE TIME

    window_size = (big_window, big_window, small_window, small_window)
    Nw = (h**2)*(small_window**2)
    padded_image = np.pad(image_array, pad_width=big_window//2, mode=mode)  # pad the image
    neighbours = find_neighbours_nlm(padded_image, small_window, big_window, *image_array.shape) 
    _filter = scipy.ndimage.generic_filter(neighbours, 
                                          _nlm, 
                                          size=window_size, 
                                          extra_arguments=(small_window, big_window, Nw,), 
                                          mode=mode)
    return _filter


def non_local_means(image_array, h, small_window, big_window, mode="reflect"):
    """Non-local means (NLM) filter 
    :param image_array: image array
    :type image_array: np.ndarray
    :param h: h parameter
    :type h: int
    :param small_window: small search window
    :type small_window: int
    :param big_window: big search window
    :type big_window: int
    :return: output image
    :rtype: np.ndarray

    """
    # refer to this paper for the implementation: 
    # https://www.researchgate.net/publication/38294293_Nonlocal_Means-Based_Speckle_Filtering_for_Ultrasound_Images

    pad_width = big_window // 2  # define pad width
    padded_image = np.pad(image_array, pad_width=pad_width, mode=mode)  # pad the image

    output_image = np.zeros(image_array.shape)  # create output array
    height, width = image_array.shape

    neighbours = find_neighbours_nlm(padded_image, small_window, big_window, height, width) # preprocessing the neighbours of each pixel
    Nw = (h**2) * (small_window**2) # calculating neighbourhood window

    for i in tqdm(range(pad_width, pad_width + height), desc="NLM user implementation", leave=True):
        for j in range(pad_width, pad_width + width):
            pixel_window = neighbours[i,j]  # (small_window x small_window) array around the target pixel 
            # (nested array inside the whole array)
            neighbour_window = neighbours[(i - pad_width):(i + pad_width + 1) , (j - pad_width):(j + pad_width + 1)]  
            # (big_window x big_window) sliced pixel neighbourhood array around the target pixel

            Ip = evaluate_norm_nlm(pixel_window, neighbour_window, Nw)  # calculating Ip
            output_image[i - pad_width, j - pad_width] = np.clip(Ip, 0, 255) # keeping the pixel values between 0-255 

    return output_image.astype("uint8")


def find_neighbours_nlm(padded_image, small_window,big_window, h, w):  # this function simply stores the 
    #  neighbour array around each pixel (including the padded area).
    small_width, big_width = small_window // 2, big_window // 2
    neighbours = np.zeros((padded_image.shape[0], padded_image.shape[1], small_window, small_window))  # make output array

    # finding the neighbors of each pixel in the original image using the padded image
    for i in range(big_width, big_width + h):
        for j in range(big_width, big_width + w):   
            neighbours[i,j] = padded_image[(i - small_width):(i + small_width + 1), (j - small_width):(j + small_width + 1)]
    
    return neighbours

# function to calculate the weighted average value (Ip) for each pixel
def evaluate_norm_nlm(pixel_window, neighbour_window, Nw):  # optimised eval function using only numpy functions instead of python loops
    squared_diff = (neighbour_window - pixel_window)**2  # find the squared difference between each neighbouring pixel array 
    # and the original pixel array around the target pixel
    w = np.exp(-1*(np.sum(squared_diff, axis=(2, 3)))/Nw)  # sum the subarrays elementwise, 
    # so each subarray is summed but not the outer array (THIS WAS HARD TO FIGURE OUT, FUNNILY ENOUGH.)

    Iq = neighbour_window[:, :, pixel_window.shape[0]//2, pixel_window.shape[1]//2]  # find all central values 'Iq' of the neighbours

    Ip = np.sum(np.multiply(w, Iq))  # calculate Ip
    Z = np.sum(w)  # calculate Z
    return Ip/Z  # return final value


def _ad(window, window_size, k, gamma):

    window = window.reshape(window_size)  # reshape as it gives the window as a 1D arr. 

    centre_x, centre_y = window.shape[0] // 2, window.shape[1] // 2

    dN = window[centre_x, centre_y - 1] - window[centre_x, centre_y]  # delta centre v north
    dS = window[centre_x, centre_y + 1] - window[centre_x, centre_y]  # delta centre v south
    dE = window[centre_x + 1, centre_y] - window[centre_x, centre_y]  # delta centre v east
    dW = window[centre_x - 1, centre_y] - window[centre_x, centre_y]  # delta centre v west
    
    gN = np.exp(-(dN / k) ** 2.)  # function defined in paper
    gS = np.exp(-(dS / k) ** 2.)
    gE = np.exp(-(dE / k) ** 2.)
    gW = np.exp(-(dW / k) ** 2.)

    return window[centre_x, centre_y] * (1 - gamma * (gN + gS + gE + gW)) + (gamma * (gN * window[centre_x, centre_y - 1] + 
                                                                                      gS * window[centre_x, centre_y + 1] + 
                                                                                      gE * window[centre_x + 1, centre_y] + 
                                                                                      gW * window[centre_x - 1, centre_y]))
   

def anisotropic_diffusion_filter(image_array, k, gamma, window_size=(3, 3), mode="reflect"):
    _filter = scipy.ndimage.generic_filter(image_array.astype("float32"), 
                                          _ad, 
                                          size=window_size, 
                                          extra_arguments=(window_size, k, gamma,), 
                                          mode=mode)
    return _filter


def _lee(window, var_k, ):

    """Refer to the paper about SRAD for this implementation."""

    mean = np.mean(window)
    w = var_k / (var_k + np.var(window))
    
    centre_value = window[window.shape[0]//2]

    return mean + w * (centre_value - mean)

def lee_filter(image_array, window_size=(3, 3), mode="reflect"):
    var_k = np.var(image_array)
    _filter = scipy.ndimage.generic_filter(image_array.astype("float32"), 
                                          _lee, 
                                          size=window_size, 
                                          extra_arguments=(var_k,), 
                                          mode=mode)
    return _filter


def find_neighbours_srad(image_array, pad_width=1, mode="reflect"):
        padded_image = np.pad(image_array, pad_width=pad_width, mode=mode)  # pad the image
        return (
            padded_image[2:, 1:-1],    # south
            padded_image[:-2, 1:-1],   # north
            padded_image[1:-1, 2:],    # east
            padded_image[1:-1, :-2],   # west
        )  # return the neighbours, refer to equation 28-29


def compute_variation_coefficient(image_array, epsilon=1e-12):
    nS, nN, nE, nW = find_neighbours_srad(image_array)  # equation 28, 29

    dS = nS - image_array
    dN = nN - image_array
    dE = nE - image_array
    dW = nW - image_array

    gradient = np.sqrt(np.multiply(dS, dS) + np.multiply(dN, dN) + np.multiply(dE, dE) + np.multiply(dW, dW))   
    # delta I  (hereon referred to as dI), equation 30.5, the proposition 
    laplacian = (nS + nN + nE + nW - 4*image_array)  # delta I^2 (hereon referred to as dI^2)

    gradient_norm = np.divide(gradient, (image_array + epsilon))  # dI/I
    laplacian_norm = np.divide(laplacian, (image_array + epsilon))  # dI^2/I

    # compute initial Q, equation 30 and 31, https://ieeexplore.ieee.org/document/1097762

    numerator = (0.5 * np.multiply(gradient_norm, gradient_norm)) - (np.multiply(laplacian_norm, laplacian_norm) / 16) 
    denominator = 1 + (0.25 * np.multiply(laplacian_norm, laplacian_norm)) 
    denominator_sqr = np.multiply(denominator, denominator)

    q = np.sqrt(np.divide(numerator, denominator_sqr))  # eq. 31

    return q


def srad(image_array, n_iters, time_step, decay_factor, mode="reflect"):
    """Speckle-Reducing Anisotropic Diffusion. Refer to this paper for the implementation:
    https://ieeexplore.ieee.org/document/1097762 

    :param image_array: image array
    :type image_array: np.ndarray
    :param n_iters: iterations to run the SRAD algorithm
    :type n_iters: int
    :param time_step: the time step of the algorithm
    :type time_step: float
    :param decay_factor: the decay factor of the SRAD algorithm
    :type decay_factor: float
    :param mode: padding mode, defaults to "reflect"
    :type mode: str, optional
    :return: output image
    :rtype: np.ndarray
    """

    # GENUINELY NIGHTMARISH IMPLEMENTATION

    image_array = image_array.astype("float32")

    nS, nN, nE, nW = find_neighbours_srad(image_array, mode=mode)  # equation 28, 29

    dS = nS - image_array
    dN = nN - image_array
    dE = nE - image_array
    dW = nW - image_array  # defining the deltas so that later 
    # we can use them in the divergence calculation.

    q = compute_variation_coefficient(image_array)

    Q_0 = q != 0
    Q_0 = Q_0.astype('float32')

    epsilon = 1e-12  # to make sure that we have no zero values we add epsilon to original image array I when dividing. 

    t = 0 # starting timestep

    for _ in tqdm(range(n_iters), desc="Running SRAD", leave=True):

        q = compute_variation_coefficient(image_array)  # recalculate q 
        q_0 = Q_0 * np.exp(-decay_factor*t)  # equation 37

        q_0_sqr = np.multiply(q_0, q_0)
        q_sqr = np.multiply(q, q)
        eq_33 = np.divide((q_sqr - q_0_sqr), (np.multiply(q_0_sqr, (1 + q_0_sqr)) + epsilon))
        c = np.exp(-eq_33)   # coefficients of diffusion, equation 33

        S_c, N_c, E_c, W_c = find_neighbours_srad(c, mode=mode)

        div = (
            S_c * dS - c * dN +
            E_c * dE - c * dW
        )  # equation 58, calculating the div coefficient

        image_array = image_array + ((time_step / 4) * div)  # equation 61
        t += time_step 
    
    return np.clip(image_array, 0, 255).astype(np.uint8)  # clip values and return as uint8.

