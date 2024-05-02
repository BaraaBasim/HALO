import cv2
import os

# Set the directory path for the images
images_dir = os.getcwd()

# Set the output directory path for the merged image
output_dir = "/home/ps/Baraa/physics_based_image_generation_model_matlab_script/HALOAttack/v0/patch/results"

# Get the list of image file names in the directory
image_files = os.listdir(images_dir)

# Filter image files only
image_files = [file for file in image_files if file.endswith((".jpg", ".jpeg", ".png"))]

# Sort the image files alphabetically
image_files.sort()

# Create an empty list to store the images
images = []
new_size = (160, 177)
# Read and append the images to the list
for file in image_files[:12]:
    image_path = os.path.join(images_dir, file)
    image = cv2.imread(image_path)
    image = cv2.resize(image, new_size)
    images.append(image)

# Create a new blank image with the size of the first image
merged_image = images[0].copy()
  
# Merge each image on top of the previous one using addWeighted
for image in images[1:]:
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)  # Wait for any key press
    # cv2.destroyAllWindows()  # Close the window
    merged_image = cv2.addWeighted(merged_image, 1, image, 0.4, 0)

# Save the merged image

output_path = os.path.join(output_dir, "adversarial_shirt.jpg")
cv2.imwrite(output_path, merged_image)

print("Merged image saved successfully!")