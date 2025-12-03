import cv2

def extract_squares(board_img):
    """Return list of 64 square images in FEN order (a8→h1)"""
    squares = []
    size = board_img.shape[0]  # assuming square image
    sq_size = size // 8

    for row in range(8):
        for col in range(8):
            y1 = row * sq_size
            x1 = col * sq_size
            square = board_img[y1:y1+sq_size, x1:x1+sq_size]
            squares.append(square)
    return squares

