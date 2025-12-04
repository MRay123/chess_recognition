import cv2
import matplotlib.pyplot as plt
import chess
import chess.svg
from IPython.display import SVG, display

def visualize_board_overlay(board_img, board_state, class_labels):
    h, w, _ = board_img.shape
    square_h, square_w = h//8, w//8
    img_copy = board_img.copy()
    for row in range(8):
        for col in range(8):
            label = class_labels[board_state[row, col]]
            if label != "--":
                x, y = col*square_w + 5, row*square_h + 25
                cv2.putText(img_copy, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,0,255), 2)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def display_board_from_fen(fen):
    board = chess.Board(fen)
    svg_board = chess.svg.board(board=board, size=400)
    display(SVG(svg_board))
    
import matplotlib.pyplot as plt

def visualize_board_overlay(board_img, board_state):
    """
    board_img: top-down board image (numpy array)
    board_state: 8x8 array of FEN symbols (strings)
    """
    img = board_img.copy()
    h, w = img.shape[:2]
    square_h = h // 8
    square_w = w // 8

    for row in range(8):
        for col in range(8):
            label = board_state[row, col]
            if label != "--":
                x = col * square_w + square_w // 2
                y = row * square_h + square_h // 2
                cv2.putText(img, label, (x-10, y+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
    
    # Convert BGR → RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

