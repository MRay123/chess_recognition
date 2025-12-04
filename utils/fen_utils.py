import numpy as np

def board_to_fen(board_state, class_labels):
    fen_rows = []
    for row in board_state:
        fen_row = ""
        empty_count = 0
        for idx in row:
            symbol = class_labels[idx]
            if symbol == "--":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += symbol
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w KQkq - 0 1"
