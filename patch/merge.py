import cv2
import os

# Set the directory path for the images
images_dir = os.getcwd()

# Set the output directory path for the merged image
output_dir = os.getcwd()

# Get the list of image file names in the directory
image_files = os.listdir(images_dir)

# Filter image files only
image_files = [file for file in image_files if file.endswith((".jpg", ".jpeg", ".png"))]

# Sort the image files alphabetically
image_files.sort()

# Create an empty list to store the images
images = []

# Read and append the images to the list
for file in image_files[:12]:
    image_path = os.path.join(images_dir, file)
    image = cv2.imread(image_path)
    images.append(image)

# Create a new blank image with the size of the first image
merged_image = images[0].copy()

# Merge each image on top of the previous one using addWeighted
for image in images[1:]:
    merged_image = cv2.addWeighted(merged_image, 1, image, 1, 0)

# Save the merged image
output_path = os.path.join(output_dir, "12_lights.jpg")
cv2.imwrite(output_path, merged_image)

print("Merged image saved successfully!")