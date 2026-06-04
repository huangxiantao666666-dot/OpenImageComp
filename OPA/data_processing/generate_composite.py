"""
Script Name: generate_composite.py
Purpose: Interactive tool for generating synthetic composite images for the OPA (Object Placement Assessment) dataset.

This script allows users to:
1. Select a foreground object and its corresponding mask from the OPA dataset (derived from COCO)
2. Select a background image from the OPA dataset
3. Manually specify the position (x, y) and size (w, h) where the foreground should be placed
4. Assign a binary label (0 = unreasonable, 1 = reasonable) to the generated composite
5. Save the composite image with a filename encoding all placement parameters

The generated composite images follow the OPA dataset format:
- Foreground and mask undergo identical geometric transformations (resize + translation)
- Alpha blending is used for compositing: composite = foreground * mask + background * (1 - mask)
- The mask represents the precise (non-rectangular) silhouette of the foreground object
- Output filename format: fgId_bgId_x_y_w_h_scale_label.jpg

This script corresponds to Section 2.1 (Composite Image Generation) of the OPA paper.
"""

from PIL import Image
import os.path
import torch
import torchvision.transforms as transforms

# ==================== PART 1: SELECT FOREGROUND OBJECT ====================
# Initialize empty paths for foreground image and its mask
fg_name = ''
mask_name = ''

# Loop until a valid foreground ID is provided by the user
while True:
    # Prompt user to enter a foreground object ID (e.g., "123456")
    fg_id = input('Please input a foreground id:')
    
    # Walk through the OPA/foreground directory to collect all category folder names
    dirnames = []
    for parent, dirname, filenames in os.walk('OPA/foreground'):
        dirnames.extend(dirname)  # Add each subdirectory name (category) to the list
    
    # Search through each category folder to find the foreground image and its mask
    for cat in dirnames:
        fg_name = os.path.join('OPA/foreground/', cat, fg_id + '.jpg')
        mask_name = os.path.join('OPA/foreground/', cat, 'mask_' + fg_id + '.jpg')
        if os.path.exists(fg_name):  # Found the foreground image
            break  # Exit the inner loop
    
    # If the foreground image was not found in any category, prompt user to re-enter
    if not os.path.exists(fg_name):
        print("This ID does not exist!")
        continue  # Restart the while loop
    else:
        break  # Valid foreground ID found, exit the outer loop


# ==================== PART 2: SELECT BACKGROUND IMAGE ====================
bg_name = ''

# Loop until a valid background ID is provided
while True:
    bg_id = input('Please input a background id:')
    
    # Walk through the OPA/background directory to collect all category folder names
    dirnames = []
    for parent, dirname, filenames in os.walk('OPA/background'):
        dirnames.extend(dirname)
    
    # Search through each category folder to find the background image
    for cat in dirnames:
        bg_name = os.path.join('OPA/background/', cat, bg_id + '.jpg')
        if os.path.exists(bg_name):
            break
    
    if not os.path.exists(bg_name):
        print("This ID does not exist!")
        continue
    else:
        break


# ==================== PART 3: LOAD IMAGES ====================
# Load foreground image as RGB (3 channels)
fg_img = Image.open(fg_name).convert('RGB')
# Load mask image as grayscale (single channel, 0=background, 1=foreground)
mask_img = Image.open(mask_name).convert('L')
# Load background image as RGB
bg_img = Image.open(bg_name).convert('RGB')


# ==================== PART 4: SPECIFY FOREGROUND POSITION AND SIZE ====================
# Get background dimensions
bg_h = bg_img.height
bg_w = bg_img.width

print("The size of the background is {} * {}. Please input the position of the foreground.".format(bg_w, bg_h))

str_scale = ''  # Will store the scale ratio as a formatted string

# Loop until valid placement coordinates are provided
while True:
    left = int(input('x:'))      # X-coordinate of the top-left corner of the foreground bounding box
    top = int(input('y:'))       # Y-coordinate of the top-left corner
    w = int(input('w:'))         # Width of the foreground bounding box (after resizing)
    right = w + left             # Calculate right boundary (exclusive)
    h = int(input('h:'))         # Height of the foreground bounding box (after resizing)
    bottom = h + top             # Calculate bottom boundary (exclusive)
    
    # Validate placement: dimensions must be positive and within background bounds
    if right - left <= 0 or bottom - top <= 0 or right > bg_w or bottom > bg_h:
        print('This position is illegal!')
        continue
    else:
        # Calculate scale ratio = max(relative width, relative height)
        # This represents the maximum proportion of the background occupied by the foreground
        scale = max(w / bg_w, h / bg_h)
        str_scale = "%.9f" % scale  # Format with 9 decimal places for precision
        print("scale=" + str_scale)
        break


# ==================== PART 5: RESIZE FOREGROUND AND MASK ====================
# Define transformation: resize to target (height, width) and convert to Tensor (range [0, 1])
fg_transform = transforms.Compose([
    transforms.Resize((bottom - top, right - left)),  # Resize to user-specified dimensions
    transforms.ToTensor(),  # Converts PIL Image to torch.Tensor, scales pixel values to [0, 1]
])

# Apply the same transformation to both foreground and mask
fg_img_ = fg_transform(fg_img)      # Resized foreground, shape: [3, h, w]
mask_img_ = fg_transform(mask_img)  # Resized mask, shape: [1, h, w] (grayscale)


# ==================== PART 6: CREATE CANVAS AND PLACE FOREGROUND ====================
# Create zero-initialized canvas tensors with the same dimensions as the background
fg_img = torch.zeros(3, bg_h, bg_w)   # Canvas for foreground (3 RGB channels)
mask_img = torch.zeros(3, bg_h, bg_w) # Canvas for mask (replicated to 3 channels for easier multiplication)

# Place the resized foreground and mask at the specified (left, top) position
# Note: The mask is non-rectangular - it preserves the exact silhouette of the foreground object
fg_img[:, top:bottom, left:right] = fg_img_[:, :, :]
mask_img[:, top:bottom, left:right] = mask_img_[:, :, :]

# Convert background image to tensor (range [0, 1])
bg_img = transforms.ToTensor()(bg_img)


# ==================== PART 7: ALPHA BLENDING COMPOSITION ====================
# Alpha blending formula: composite = foreground * mask + background * (1 - mask)
# - Where mask = 1: show foreground
# - Where mask = 0: show background
# - Where mask is between 0 and 1 (at edges): blend foreground and background
blended = fg_img * mask_img + bg_img * (1 - mask_img)

# Convert the blended tensor back to a PIL Image
com_pic = transforms.ToPILImage()(blended).convert('RGB')
# Uncomment the line below to display the composite image:
# com_pic.show()


# ==================== PART 8: INPUT LABEL AND SAVE PATH ====================
# Loop until a valid label (0 or 1) is provided
while True:
    label = input("Please input a label:")  # 0 = unreasonable placement, 1 = reasonable placement
    if label != '0' and label != '1':
        print('This label is illegal!')
        continue
    else:
        break

# Loop until a valid save directory path is provided
while True:
    save_path = input('Please input a path to save your composite image:')
    if not os.path.exists(save_path):
        print('This path does not exist!')
        continue
    else:
        break


# ==================== PART 9: SAVE COMPOSITE IMAGE ====================
# Generate filename following OPA dataset convention:
# Format: fgId_bgId_x_y_w_h_scale_label.jpg
com_pic_name = fg_id + "_" + bg_id + "_" + str(left) + "_" + str(top) + "_" + str(w) + "_" + str(
    h) + "_" + '%.4f' % eval(str_scale) + "_" + label + ".jpg"

# Construct full save path and save the image
save_path = os.path.join(save_path, com_pic_name)
com_pic.save(save_path)