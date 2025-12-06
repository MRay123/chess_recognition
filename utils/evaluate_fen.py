import os
import re
from typing import Tuple


def extract_fen_from_filename(filename: str) -> str:


    name = os.path.splitext(filename)[0]
    parts = name.split(" ")


    board = parts[0].replace("-", "/").replace("_", "/")

    
    side = parts[1] if len(parts) > 1 else "w"      
    castling = parts[2] if len(parts) > 2 else "-"  
    ep = parts[3] if len(parts) > 3 else "-"        
    halfmove = parts[4] if len(parts) > 4 else "0"
    fullmove = parts[5] if len(parts) > 5 else "1"

    fen = f"{board} {side} {castling} {ep} {halfmove} {fullmove}"
    return fen


def fen_to_board_array(fen: str):
    board = fen.split(" ")[0]
    ranks = board.split("/")

    squares = []
    for r in ranks:
        for ch in r:
            if ch.isdigit():
                squares.extend(["."] * int(ch))
            else:
                squares.append(ch)
    return squares  


def compare_fens(true_fen: str, predicted_fen: str):
    true_arr = fen_to_board_array(true_fen)
    pred_arr = fen_to_board_array(predicted_fen)

    assert len(true_arr) == len(pred_arr) == 64

    correct = sum(1 for t, p in zip(true_arr, pred_arr) if t == p)
    accuracy = correct / 64.0

    mismatches = [
        (i, true_arr[i], pred_arr[i])
        for i in range(64)
        if true_arr[i] != pred_arr[i]
    ]

    return accuracy, mismatches


def evaluate_image(filename: str, predicted_fen: str):

    true_fen = extract_fen_from_filename(os.path.basename(filename))
    board_acc, mismatches = compare_fens(true_fen, predicted_fen)

    print("\n=== FEN EVALUATION ===")
    print("True FEN:     ", true_fen)
    print("Predicted FEN:", predicted_fen)
    print(f"Board Accuracy: {board_acc*100:.2f}%")

    if true_fen == predicted_fen:
        print("✔ Full FEN MATCH")
    else:
        print("✘ Full FEN mismatch")

    if mismatches:
        print("\nSquare mismatches:")
        for idx, t, p in mismatches:
            rank = 8 - (idx // 8)
            file = "abcdefgh"[idx % 8]
            print(f"  {file}{rank}: expected {t}, got {p}")
    else:
        print("No mismatches. Perfect board.")



if __name__ == "__main__":
    
    filename = "sample_boards/rnbqkbnr_pppppppp_8_8_8_8_PPPPPPPP_RNBQKBNR w KQkq - 0 1.png"

    
    predicted_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    evaluate_image(filename, predicted_fen)
