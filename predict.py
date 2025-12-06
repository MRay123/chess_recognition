import cv2
import torch
from torchvision import models, transforms, datasets
from utils.preproccess import extract_squares_from_board
from utils.visualize import visualize_board_overlay, display_board_from_fen, visualize_board_overlay
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.ImageFolder("data/train")
idx_to_folder = {v: k for k,v in train_data.class_to_idx.items()}

# Map folder names to FEN symbols
folder_to_fen = {
    "empty": "--",
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k"
}


num_classes = 13
model = models.resnet18(weights=None)  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("models/chess_square_resnet18_v3_epoch20.pth", map_location=device))
model = model.to(device)
model.eval()


transform_square = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


def board_to_fen(board_state):
    """Convert 8x8 array of FEN symbols to a FEN string."""
    fen_rows = []
    for row in board_state:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "--":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def predict_board_from_image(img_path):
    
    squares, top_down_board = extract_squares_from_board(img_path)
    
    
    board_state = []
    for sq in squares:
        sq_rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        sq_pil = Image.fromarray(sq_rgb)
        
        inp = transform_square(sq_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(inp)
            _, pred = torch.max(out,1)
        
        pred_folder_name = idx_to_folder[pred.item()]
        pred_symbol = folder_to_fen[pred_folder_name]
        board_state.append(pred_symbol)
    
    board_state = np.array(board_state).reshape(8,8)
    fen = board_to_fen(board_state)
    
    return board_state, fen, top_down_board

def predict_board_fen(img_path: str) -> str:
    board_state, fen, top_down_board = predict_board_from_image(img_path)
    return fen


if __name__ == "__main__":
    img_path = "data/sample_boards/1K6-8-4q3-1nP3k1-4r1B1-1q1n2q1-2RB4-1q1r4.jpeg"
    board_state, fen, top_down_board = predict_board_from_image(img_path)
    
    print("Predicted FEN:", fen)
    display_board_from_fen(fen)
    visualize_board_overlay(top_down_board, board_state)
