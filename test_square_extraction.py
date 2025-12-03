import cv2
import os
import matplotlib.pyplot as plt
from board_detection import detect_board_corners, warp_board
from square_extraction import extract_squares

# -------------------------
# Load image
# -------------------------
image_path = "data/test/1.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# -------------------------
# Detect board corners
# -------------------------
corners = detect_board_corners(image)

if corners is None:
    raise ValueError("❌ Board not detected")

# -------------------------
# Warp the board
# -------------------------
warped = warp_board(image, corners, out_size=800)

# -------------------------
# Draw simple gridlines
# -------------------------
grid_img = warped.copy()
sq_size = warped.shape[0] // 8

for i in range(1, 8):
    cv2.line(grid_img, (0, i * sq_size), (warped.shape[1], i * sq_size), (0, 255, 0), 2)
    cv2.line(grid_img, (i * sq_size, 0), (i * sq_size, warped.shape[0]), (0, 255, 0), 2)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
plt.title("Warped Board with Gridlines")
plt.axis("off")
plt.show()

# -------------------------
# Extract squares
# -------------------------
squares = extract_squares(warped)

# -------------------------
# Save squares
# -------------------------
output_dir = "outputs/squares_test"
os.makedirs(output_dir, exist_ok=True)

for i, sq in enumerate(squares):
    cv2.imwrite(f"{output_dir}/square_{i}.png", sq)

# -------------------------
# Show first 16 squares
# -------------------------
plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(cv2.cvtColor(squares[i], cv2.COLOR_BGR2RGB))
    plt.axis("off")
plt.show()
