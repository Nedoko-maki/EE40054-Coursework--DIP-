import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import scipy.ndimage


def image_size(image_data: np.ndarray):
    print(f"Image is of shape {image_data.shape}, size {image_data.size}, max value {np.max(image_data)}")
    return image_data.shape


def conv_dec_to_bin(dec): # convert decimal value to a binary value
    pass


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


def _weighted_median(window, weights):
    window = np.multiply(window, weights)
    return np.median(window)

def weighted_median_filter(image_array, weights, window_size=(3, 3), mode="reflect"):
    return scipy.ndimage.generic_filter(image_array, 
                                        _weighted_median, 
                                        size=window_size, 
                                        extra_arguments=(weights,), 
                                        mode=mode)


def _adaptive_median(window, window_size, shift, centre_weight, constant):
    positions = np.argwhere(np.full(window_size, 1))  # make a 1-filled matrix with the size of the window, 
    # and get the positions of the elements. 
    distance = np.array([((x[0]-shift)**2 + (x[1]-shift)**2)**(0.5) for x in positions]).reshape(window_size)
    # shift the positions so that 0, 0 is in the centre of the window, and get the distances from 0, 0. 

    k = (constant * np.std(window)) / np.mean(window)  # get a constant to reduce the matrix operations needed
    # this doesn't check for zero mean division

    weights = np.full(window_size, centre_weight)
    weights = np.subtract(weights, np.multiply(np.full(window_size, k), distance))

    return _weighted_median(window.reshape(window_size), weights)  # pass off the final calculation to the weighted median f. 

def adaptive_median_filter(image_array, centre_weight, constant, window_size=(3, 3), mode="reflect"):
    shift = window_size[0] // 2  # find the necessary shift for the centre of the window, floored div 2. 
    return scipy.ndimage.generic_filter(image_array, 
                                        _adaptive_median, 
                                        window_size, 
                                        extra_arguments=(window_size, shift, centre_weight, constant), 
                                        mode=mode)


def _geometric_mean(window, k):
    return np.power(np.prod(window), k) 

def geometric_mean_filter(image_array, window_size=(3, 3), mode="reflect"):  # most commonly used to filter out Gaussian noise
    k = 1.0/np.prod(window_size)  # save a bit of processing time by preprocessing the weight.
    filter = scipy.ndimage.generic_filter(image_array, 
                                          _geometric_mean, 
                                          window_size, 
                                          extra_arguments=(k,),
                                          mode=mode)
    return filter


def _arithmetic_mean(window, window_prod):
    return np.sum(window) / (window_prod)

def arithmetic_mean_filter(image_array, window_size=(3, 3), mode="reflect"):
    _window_prod = np.prod(window_size)  # save a bit of processing time by preprocessing the product of the window dims.
    filter = scipy.ndimage.generic_filter(image_array, 
                                          _arithmetic_mean, 
                                          window_size, 
                                          extra_arguments=(_window_prod,), 
                                          mode=mode)
    return filter


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


def histogram_equalisation(image_array):
    hist, _ = np.histogram(image_array.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    # cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    new_image_array = cdf[image_array]

    return new_image_array


def gaussian_filter(image_array, sigma=1, mode="reflect"):
    return scipy.ndimage.gaussian_filter(image_array, sigma, mode=mode)


# gaussian lowpass filter

