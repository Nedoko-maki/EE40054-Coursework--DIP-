from pathlib import Path
from image_processing import (bit_plane_slicing, 
                              image_size,
                              mean_filter, 
                              trimmed_mean_filter,
                              median_filter,
                              weighted_median_filter,
                              adaptive_median_filter,
                              _adaptive_median,
                              geometric_mean_filter,
                              arithmetic_mean_filter,
                              histogram_equalisation,
                              histogram_representation,
                              gaussian_filter)  # import all the filters to be tested
from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt

class ImageProcessor:  
    # custom class to keep some variables together, add filters and their args, and to process the image
    def __init__(self, image_array):
        self.im = image_array
        self.orig_im = image_array
        self.filters = []

    def add_filter(self, filter, *args, **kwargs):  # stores the filter function, their args and kwargs in a list.
        self.filters.append((filter, args, kwargs))
    
    def process_image(self, verbose=False):  # applies the filter functions in the stored list 'filters' sequentially. 
        for f, args, kwargs in self.filters:
            
            if verbose:
                print(f"Processing image with filter {f.__name__}")

            self.im = f(self.im, *args, **kwargs)

        return self.im


def read_image(filepath: Path):
    # match filepath.suffix:
    #     case ".png":
    #         pass

    #     case ".jpg":
    #         pass

    #     case ".pgm":
    #         pass
    
    image = Image.open(filepath)
    image_array = np.asarray(image)

    return image_array 


def display_image(image_data):  # show the images side by side. 
    
    if isinstance(image_data, (list, tuple)):
        fig, ax = plt.subplots(nrows=1, ncols=len(image_data))

        for i, axi in enumerate(ax.flat):
            axi.imshow(image_data[i], alpha=1)
    else:
        imgplot = plt.imshow(image_data)
    
    plt.show()  # show the images


def main():
    images = []

    image_filepath = Path("images/NZjers1.png")  # image filepath
    # image_filepath = Path("images/foetus.png")  # image filepath

    image_array = read_image(image_filepath)  # read the image and save into a numpy array
    # image_size(image_array)  # test function for displaying imsize
    images.append(image_array)  # store the original image in a list

    # histogram_representation(stage_2.im)

    for h, tws, sws in [(18, 7, 21),]:
        img = ImageProcessor(image_array)
        img.add_filter(cv2.fastNlMeansDenoising, h=h, templateWindowSize=tws, searchWindowSize=sws)
        img.add_filter(mean_filter)
        img.process_image(verbose=True)
        images.append(img.im)


    img_canny = ImageProcessor(images[-1])
    img_canny.add_filter(cv2.Canny, 230, 500)
    img_canny.process_image()
    images.append(img_canny.im)
    # h=18, tws=7, sws=21

    # cv2_impl = ImageProcessor(image_array)
    # cv2_impl.add_filter(cv2.fastNlMeansDenoising, h=15, templateWindowSize=7, searchWindowSize=21)
    # cv2_impl.process_image()


    display_image(images)  # display the images for easy review. 

    # ret = bit_plane_slicing(image_array)
    # for i in ret:
    #     i
    #     display_image(i)



if __name__ == "__main__":
    main()