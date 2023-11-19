import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def mean_filter(image, kernel_size=3, padding=1):
    """
    Apply a mean filter to an image manually without using convolve2d.

    :param image: Input image as a numpy array.
    :param kernel_size: Size of the mean filter kernel.
    :param padding: Padding applied to the image.
    :return: Filtered image as a numpy array.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Pad the image
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    # Initialize the output image
    height, width = image.shape
    output = np.zeros((height, width))

    # Calculate the offset for the kernel
    offset = kernel_size // 2

    # Iterate over each pixel in the image
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Calculate the average value within the kernel's neighborhood
            sum_value = 0.0
            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    sum_value += image[y + ky, x + kx]
            average = sum_value / (kernel_size ** 2)
            output[y, x] = average

    # Remove the padding
    output = output[offset:height - offset, offset:width - offset]

    return output

def manual_median_filter(image, kernel_size=3, padding=1):
    """
    Apply a median filter to an image manually.

    :param image: Input image as a numpy array.
    :param kernel_size: Size of the median filter kernel.
    :param padding: Padding applied to the image.
    :return: Filtered image as a numpy array.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Pad the image
    if padding > 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    # Initialize the output image
    height, width = image.shape
    output = np.zeros((height, width))

    # Calculate the offset for the kernel
    offset = kernel_size // 2

    # Iterate over each pixel in the image
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Collect the values within the kernel's neighborhood
            neighborhood = []
            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    neighborhood.append(image[y + ky, x + kx])
            
            # Find the median value
            median_value = np.median(neighborhood)
            output[y, x] = median_value

    # Remove the padding
    output = output[offset:height - offset, offset:width - offset]

    return output

# Function to create and save histograms using Matplotlib
def create_and_save_histogram(image, filename, title):
    if len(image.shape) == 3:
        image = image.mean(axis=2).astype(np.uint8)

    plt.figure()
    plt.hist(image.flatten(), bins=256, range=[0, 256], color='black')
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(filename)
    plt.close()

# Load the images
img_path1 = 'test_img/noise1.png'
img_path2 = 'test_img/noise2.png'

img1 = Image.open(img_path1)
img2 = Image.open(img_path2)

# Convert images to numpy arrays
img1_np = np.array(img1)
img2_np = np.array(img2)

# Apply the mean filter
filtered_img1 = mean_filter(img1_np)
filtered_img2 = mean_filter(img2_np)

# Apply the median filter
median_filtered_img1 = manual_median_filter(img1_np)
median_filtered_img2 = manual_median_filter(img2_np)

# Convert the filtered images back to PIL images for saving and displaying
filtered_img1_pil = Image.fromarray(filtered_img1.astype('uint8'))
filtered_img2_pil = Image.fromarray(filtered_img2.astype('uint8'))
median_filtered_img1_pil = Image.fromarray(median_filtered_img1.astype('uint8'))
median_filtered_img2_pil = Image.fromarray(median_filtered_img2.astype('uint8'))

# Save the filtered images
filtered_img1_path = 'result_img/noise1_q1.png'
filtered_img2_path = 'result_img/noise2_q1.png'
median_filtered_img1_path = 'result_img/noise1_q2.png'
median_filtered_img2_path = 'result_img/noise2_q2.png'

filtered_img1_pil.save(filtered_img1_path)
filtered_img2_pil.save(filtered_img2_path)
median_filtered_img1_pil.save(median_filtered_img1_path)
median_filtered_img2_pil.save(median_filtered_img2_path)

# Paths to save the histogram images
histogram_filenames = {
    "noise1_his": "result_img/noise1_his.png",
    "noise1_q1_his": "result_img/noise1_q1_his.png",
    "noise1_q2_his": "result_img/noise1_q2_his.png",
    "noise2_his": "result_img/noise2_his.png",
    "noise2_q1_his": "result_img/noise2_q1_his.png",
    "noise2_q2_his": "result_img/noise2_q2_his.png"
}

# Generate and save histograms
titles = {
    "noise1_his": "Histogram of Noise 1",
    "noise1_q1_his": "Histogram of Noise 1 (Mean Filter)",
    "noise1_q2_his": "Histogram of Noise 1 (Median Filter)",
    "noise2_his": "Histogram of Noise 2",
    "noise2_q1_his": "Histogram of Noise 2 (Mean Filter)",
    "noise2_q2_his": "Histogram of Noise 2 (Median Filter)"
}
for key, filename in histogram_filenames.items():
    image = img1_np if 'noise1' in key else img2_np
    if 'q1' in key:
        image = filtered_img1 if 'noise1' in key else filtered_img2
    elif 'q2' in key:
        image = median_filtered_img1 if 'noise1' in key else median_filtered_img2
    create_and_save_histogram(image, filename, titles[key])