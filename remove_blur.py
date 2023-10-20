import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def blur_remove(folder_name):
  # Go through the folder of images.
  print("\n\n================================================================================================")
  print(f"              Removing blurry images:   {folder_name} ")
  print("================================================================================================\n")
  image_paths = []
  for filename in os.listdir(folder_name):
    file_path = os.path.join(folder_name, filename)

    if os.path.isfile(file_path):
      image_paths.append(file_path)

  # Calculate the Laplacian and Laplacian variance of all images to detect which images are blurry.
  laplacian_variances = []
  for image_path in image_paths:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    laplacian_variances.append(laplacian_variance)

  # Calculate the mean and standard deviation of the Laplacian variance.
  mean_laplacian_variance = np.mean(laplacian_variances)
  std_laplacian_variance = np.std(laplacian_variances)
  
  # Plot a histogram showing the Laplacian variance on the x-axis and frequency on the y-axis.
  plt.hist(laplacian_variances)
  plt.xlabel('Laplacian Variance')
  plt.ylabel('Frequency')
  plt.title('Laplacian Variance Histogram')
  # Save the histogram outside the images folder.
  histogram_path = os.path.join(folder_name, 'laplacian_variance_histogram.png')
  plt.savefig(histogram_path)
  plt.show()
    # Calculate the threshold for blurry images.
  if(mean_laplacian_variance< std_laplacian_variance):
    print("Dataset is skewed")
    threshold = mean_laplacian_variance - std_laplacian_variance/2.5
  else:
    threshold = mean_laplacian_variance - std_laplacian_variance
    print("Dataset is normal")

  threshold = threshold + 30 # aggresive image threshold
  print(f"Mean Laplacian Variance: {mean_laplacian_variance}")
  print(f"Standard Deviation: {std_laplacian_variance}")
  print(f"Threshold: {threshold}")
    # Delete all blurry images.
  blurry_image_paths = []
  for image_path in image_paths:
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_variance < threshold:
        blurry_image_paths.append(image_path)
        print("Deleting: "+image_path)
        os.remove(image_path)


if __name__ == '__main__':
  # Example usage:

  folder_name = 'new_check_images/images'
  

  blur_remove(folder_name)