import cv2
import numpy as np

# -------------------------------
# Split a perfectly aligned board into 64 squares
# -------------------------------
def split_board_into_squares(board_img):
    """
    Split a top-down board image into 64 squares.
    """
    h, w, _ = board_img.shape
    square_h, square_w = h // 8, w // 8
    squares = []
    for row in range(8):
        for col in range(8):
            y1, y2 = row * square_h, (row + 1) * square_h
            x1, x2 = col * square_w, (col + 1) * square_w
            squares.append(board_img[y1:y2, x1:x2])
    return squares

# -------------------------------
# For chess.com images
# -------------------------------
def extract_squares_from_board(img_path, board_crop=None):
    """
    Load a chess.com-style board image, crop if needed, and split into 64 squares.
    Returns:
        squares: list of 64 images
        top_down_board: the full cropped board image
    Parameters:
        img_path: path to image
        board_crop: optional tuple (x, y, w, h) to crop the board region
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at: {img_path}")

    # Crop the board region if provided
    if board_crop is not None:
        x, y, w, h = board_crop
        top_down_board = img[y:y+h, x:x+w]
    else:
        top_down_board = img.copy()  # assume full image is the board

    squares = split_board_into_squares(top_down_board)
    return squares, top_down_board
