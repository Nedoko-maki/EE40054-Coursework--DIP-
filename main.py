from pathlib import Path
import image_processing as ip
from tqdm import tqdm

from PIL import Image
import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

import matplotlib.pyplot as plt


class ImageProcessor:  
    # custom class to keep some variables together, add filters and their args, and to process the image
    def __init__(self, image_array):
        self.im = image_array
        self.orig_im = image_array
        self.filters = []

    def add_filter(self, filter, *args, enforce_uint8=False, **kwargs):  # stores the filter function, their args and kwargs in a list.
        self.filters.append((filter, args, kwargs))

        if enforce_uint8:
            self.enforce_uint8()

    def enforce_uint8(self, immediate=False):
        if immediate:
            self.im = self.im.astype("uint8")
        else:  
            self.filters.append(("enforce_uint8","",""))

    def fft(self):
        self.im = np.fft.fft2(self.im)

    def ifft(self):
        self.im = np.fft.ifft2(self.im)

    def __len__(self):
        return 1

    def process_image(self):  # applies the filter functions in the stored list 'filters' sequentially. 

        pbar = tqdm(self.filters, desc='Processing image with filter: ', leave=True)

        for f, args, kwargs in pbar:
            if f == "enforce_uint8":
                self.im = self.im.astype("uint8")
                continue

            pbar.set_description(f"Processing image with filter: {f.__name__}")
            self.im = f(self.im, *args, **kwargs)
        return self.im


def read_image(filepath: Path):
    
    image = Image.open(filepath)
    image_array = np.asarray(image)

    return image_array 


def display_images(image_data, rows, cols):  # show the images side by side. 
    
    fig = plt.figure()
    ax = []

    for data, i in zip(image_data, range(cols*rows)):
        # create subplot and append to ax
        stats = {}

        if len(data) == 1 or isinstance(data, np.ndarray):  # setting the titles and data
            title = ""
            image = data
        else:
            image, title, *_ = data

            if _:
                stats = _[0] 

        if isinstance(image, ImageProcessor):  # unwrap if direct
            image = image.im    
    
        ax.append(fig.add_subplot(rows, cols, i+1))
        ax[-1].set_title(str(title))  # set title
        ax[-1].set_xticks([])  # clear ticks
        ax[-1].set_yticks([])

        if stats:  # do some comparisons with SSIM and MSE, not great as we don't have a perfect noiseless copy to compare to. 
            mse_1, mse_2, ssim_1, ssim_2 = *stats["mse"], *stats["ssim"]
            mse = mean_squared_error(mse_1, mse_2)
            _ssim = ssim(ssim_1, ssim_2, data_range=ssim_2.max() - ssim_2.min())
            ax[-1].set_xlabel(f"MSE: {mse:.2f}, SSIM: {_ssim:.2f}")

        plt.imshow(image, alpha=1)

    plt.show()  # show the images


def generate_edges(image_data, thr1=230, thr2=500):
    img_canny = ImageProcessor(image_data)
    img_canny.add_filter(cv2.Canny, thr1, thr2)
    img_canny.process_image()
    return img_canny.im


def test_nzjers():
    
    image_filepath = Path("images/NZjers1.png")  # image filepath
    original_image = read_image(image_filepath)  # read the image and save into a numpy array
    images = [[original_image, "original image "]]  # keep images and metadata in list to display at the end

    w = np.full((5, 5), 0.1)  # weights array
    w[1:4, 1:4] = 0.1 
    w[2, 2] = 1.
    
    test_filters = [
        (ip.mean_filter, (), {"window_size": (3,3)}),
        (ip.median_filter, (), {}),
        (ip.arithmetic_mean_filter, (), {}),
        (ip.trimmed_mean_filter, (), {}),
        (ip.weighted_median_filter, (), {"window_size": (5, 5), "weights": w}),
        (ip.adaptive_median_filter, (1, 0.5), {}),
        (ip.threshold_low_filter, (50,), {}),
        (ip.non_local_means, (), {"h":17, "small_window":7, "big_window": 21}),
                    ]

    for _filter, args, kwargs in test_filters:
        im = ImageProcessor(original_image.copy())
        im.add_filter(_filter, *args, **kwargs)
        im.process_image()

        ksize = (3, 3) if "window_size" not in kwargs else kwargs["window_size"]
        images.append([im.im, f"{_filter.__name__}, kernel size: {ksize}"])

    display_images(images, rows=3, cols=3)


def main_nzjers():
    images = []  # keep images and metadata in list to display at the end
    image_filepath = Path("images/NZjers1.png")  # image filepath
    original_image = read_image(image_filepath)  # read the image and save into a numpy array

    nonlocal_image = ImageProcessor(original_image.copy())
    nonlocal_image.add_filter(ip.non_local_means, h=17, small_window=7, big_window=21)  # filters are added and run with .process_image()
    # nonlocal_image.add_filter(ip.adaptive_median_filter, 1, 0.001)
    nonlocal_image.add_filter(ip.threshold_low_filter, minVal=40)  # the args and kwargs are passed onto the filter functions. 
    
    nonlocal_image.process_image()
    
    w = np.full((5, 5), 0.1)
    w[1:4, 1:4] = 0.1 
    w[2, 2] = 1.
    # print(w)

    gaussian_image = ImageProcessor(nonlocal_image.im)
    gaussian_image.add_filter(ip.srad, 200, 0.2, 0.75)
    
    gaussian_image.add_filter(ip.gaussian_filter, sigma=0.95)
    gaussian_image.process_image()

    thresholded_image = gaussian_image.im.copy()
    thresholded_image[30 < thresholded_image] = 255  # everything above 30 is considered part of the image

    images.append([original_image, "Original image", {"mse": [original_image, original_image], "ssim": [original_image, original_image]}])
    images.append([nonlocal_image, "NLM-filtered image + thresholding (pixel < 40 = 0)", {"mse": [original_image, nonlocal_image.im], "ssim": [original_image, nonlocal_image.im]}])
    images.append([gaussian_image, "SRAD + Gaussian blurred image", {"mse": [original_image, gaussian_image.im], "ssim": [original_image, gaussian_image.im]}])
    images.append([thresholded_image, "Thresholded image (pixel > 30 = 255)"])
    images.append([generate_edges(thresholded_image, thr1=80, thr2=500), "Canny edge detection"])

    tmp = ImageProcessor(original_image)
    tmp.add_filter(ip.non_local_means, h=17, small_window=7, big_window=21)  # filters are added and run with .process_image()
    tmp.add_filter(ip.adaptive_median_filter, 1, 0.015)
    tmp.add_filter(ip.threshold_low_filter, minVal=40)  # the args and kwargs are passed onto the filter functions. 
    tmp.add_filter(ip.gaussian_filter, sigma=0.95)
    tmp.process_image()
    tmp.im[30 < tmp.im] = 255
    images.append([generate_edges(tmp.im, thr1=80), "with adaptive median filter after NLM,\n no SRAD"])

    display_images(images, rows=2, cols=3)  # display the images for easy review. 


def test_foetus():
    image_filepath = Path("images/foetus.png")  # image filepath
    original_image = read_image(image_filepath)  # read the image and save into a numpy array
    images = [[original_image, "original image "]]  # keep images and metadata in list to display at the end

    w = np.full((5, 5), 0.1)  # weights array
    w[1:4, 1:4] = 0.1 
    w[2, 2] = 1.
    
    test_filters = [
        (ip.mean_filter, (), {"window_size": (3,3)}),
        (ip.median_filter, (), {}),
        (ip.arithmetic_mean_filter, (), {}),
        (ip.trimmed_mean_filter, (), {}),
        (ip.weighted_median_filter, (), {"window_size": (5, 5), "weights": w}),
        (ip.adaptive_median_filter, (1, 0.5), {}),
        (ip.threshold_low_filter, (50,), {}),
        (ip.non_local_means, (), {"h":17, "small_window":7, "big_window": 21}),
                    ]

    for _filter, args, kwargs in test_filters:
        im = ImageProcessor(original_image.copy())
        im.add_filter(ip.histogram_equalisation)
        im.add_filter(_filter, *args, **kwargs)
        im.process_image()

        ksize = (3, 3) if "window_size" not in kwargs else kwargs["window_size"]
        images.append([im.im, f"{_filter.__name__}, kernel size: {ksize}"])

    display_images(images, rows=3, cols=3)



def main_foetus():
    images = []
    image_filepath = Path("images/foetus.png")  # image filepath
    original_image = read_image(image_filepath)
    images.append([original_image, "original image"])

    test = ImageProcessor(original_image.copy())

    
    test.add_filter(ip.non_local_means, h=17, small_window=7, big_window=21)
    test.add_filter(ip.srad, 200, 0.2, 0.75)
    # test.add_filter(ip.gaussian_filter, sigma=0.7)
    # for i in range(10):  # niter of anisotropic diff. 
    #     test.add_filter(ip.anisotropic_diffusion_filter, 10, 0.25)
    
    test.process_image()
    test.enforce_uint8(immediate=True)


    test2 = ImageProcessor(original_image.copy())
    test2.add_filter(ip.non_local_means, h=17, small_window=7, big_window=21)
    test2.process_image()
    images.append([test2.im, "NLM filter (h:17, small:7, big:21)"])
    # images.append(generate_edges(test.im, thr1=20, thr2=70))
    images.append([test.im, "SRAD filtered image\n + NLM filter (h:17, small:7, big:21)"])
    images.append([generate_edges(test2.im, thr1=40, thr2=70), "Canny edge detection, min: 40, max: 70"])
    
    display_images(images, rows=2, cols=2)


if __name__ == "__main__":
    # main_nzjers()
    # test_nzjers()
    # test_foetus()
    main_foetus()