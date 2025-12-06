import os
import sys
import csv
from typing import Tuple, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.evaluate_fen import extract_fen_from_filename, compare_fens
from predict import predict_board_fen


def predict_fen_from_image(img_path: str) -> str:
    return predict_board_fen(img_path)


def evaluate_image(img_path: str) -> Tuple[float, bool, List[Tuple]]:
    filename = os.path.basename(img_path)

    true_fen = extract_fen_from_filename(filename)
    predicted_fen = predict_fen_from_image(img_path)

    board_acc, mismatches = compare_fens(true_fen, predicted_fen)
    full_match = (board_acc == 1.0)  # ACC = 100%


    return board_acc, full_match, mismatches


def evaluate_folder(folder: str, output_csv="evaluation_results.csv"):
    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        raise RuntimeError("No images found in the folder!")

    results = []
    total_boards = len(image_files)
    total_full_matches = 0
    total_accuracy = 0.0

    print(f"\nEvaluating {total_boards} boards...\n")

    for idx, filename in enumerate(image_files, 1):
        path = os.path.join(folder, filename)

        board_acc, full_match, mismatches = evaluate_image(path)
        
        if board_acc == 100.00:
            full_match = True
    
        total_accuracy += board_acc
        total_full_matches += int(full_match)

        results.append({
            "filename": filename,
            "board_accuracy": round(board_acc, 4),
            "full_fen_match": full_match,
            "num_mismatches": len(mismatches),
        })

        print(f"[{idx}/{total_boards}] {filename}: "
              f"Acc={board_acc*100:.2f}%, FullMatch={full_match}")


    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "board_accuracy", "full_fen_match", "num_mismatches"],
        )
        writer.writeheader()
        writer.writerows(results)

    avg_acc = total_accuracy / total_boards

    print("\n=============================")
    print("         Summary")
    print("=============================")
    print(f"Boards evaluated:      {total_boards}")
    print(f"Avg board accuracy:    {avg_acc*100:.2f}%")
    print(f"Full FEN matches:      {total_full_matches}/{total_boards}")
    print(f"CSV saved to:          {output_csv}")
    print("=============================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate FEN prediction accuracy on a folder of boards.")
    parser.add_argument("folder", help="Folder containing test board images")
    parser.add_argument("--csv", default="evaluation_results.csv")
    args = parser.parse_args()

    evaluate_folder(args.folder, args.csv)
