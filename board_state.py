"""Converts square predictions to FEN and tracks game state with python-chess.

Conventions:
- `predictions` is a list of 64 labels (e.g. "wk", "bp", "empty"), row-major
  from the top-left of the warped board image.
- The board is assumed to be oriented with white at the bottom, so row 0 of
  the predictions is rank 8 (black's back rank).
"""

import datetime
import os

import chess
import chess.pgn

# Square labels for the standard starting position, row-major from rank 8.
# Note this contains every one of the 13 classes - which is what makes
# auto-calibration possible (see calibration.py).
STARTING_PREDICTIONS = (
    ["br", "bn", "bb", "bq", "bk", "bb", "bn", "br"]
    + ["bp"] * 8
    + ["empty"] * 32
    + ["wp"] * 8
    + ["wr", "wn", "wb", "wq", "wk", "wb", "wn", "wr"]
)


# Our labels are color + piece letter ("wk", "bn"). FEN uses uppercase for
# white, lowercase for black, with the same piece letters.
def _label_to_fen_char(label):
    color, piece = label[0], label[1]
    return piece.upper() if color == "w" else piece


def predictions_to_placement(predictions):
    """Converts 64 square labels to a FEN piece-placement string.

    Returns e.g. "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" for the
    starting position. This is only the first field of a full FEN - turn,
    castling rights etc. are tracked by GameTracker, since they can't be
    seen in a photo.
    """
    if len(predictions) != 64:
        raise ValueError(f"Expected 64 predictions, got {len(predictions)}")

    ranks = []
    for r in range(8):  # row 0 = rank 8
        rank_str = ""
        empty_run = 0
        for c in range(8):
            label = predictions[r * 8 + c]
            if label == "empty":
                empty_run += 1
            else:
                if empty_run:
                    rank_str += str(empty_run)
                    empty_run = 0
                rank_str += _label_to_fen_char(label)
        if empty_run:
            rank_str += str(empty_run)
        ranks.append(rank_str)
    return "/".join(ranks)


class GameTracker:
    """Maintains a chess.Board and infers moves from successive board readouts.

    Instead of diffing squares (which breaks on castling and captures), each
    update searches the legal moves of the current position for one that
    produces the detected piece placement. A readout that matches no legal
    move leaves the game state untouched, so misclassified frames can't
    corrupt the game.
    """

    START_PLACEMENT = chess.Board().board_fen()

    def __init__(self):
        self.board = chess.Board()
        self.moves = []  # SAN history

    @property
    def fen(self):
        """Full FEN of the current tracked position."""
        return self.board.fen()

    def update(self, predictions):
        """Feeds a new board readout. Returns a status dict:

        status: "no_change"   - readout matches the current position
                "move"        - a legal move was detected (key "san" has it)
                "new_game"    - readout is the starting position; tracker reset
                "unrecognized" - readout matches no legal move (noise, a hand
                                 over the board, or several moves were missed)
        """
        placement = predictions_to_placement(predictions)

        if placement == self.board.board_fen():
            return {"status": "no_change"}

        if placement == self.START_PLACEMENT:
            self.board.reset()
            self.moves = []
            return {"status": "new_game"}

        for move in self.board.legal_moves:
            san = self.board.san(move)
            self.board.push(move)
            if self.board.board_fen() == placement:
                self.moves.append(san)
                return {"status": "move", "san": san, "move": move}
            self.board.pop()

        return {"status": "unrecognized", "placement": placement}

    def to_pgn(self, event="Chess CV live capture", white="White", black="Black"):
        """Returns the tracked game as a PGN string."""
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = event
        game.headers["Site"] = "Chess CV"
        game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
        game.headers["White"] = white
        game.headers["Black"] = black
        return str(game)

    def save_pgn(self, directory="games"):
        """Writes the game to a timestamped .pgn file. Returns the path."""
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"game_{stamp}.pgn")
        with open(path, "w") as f:
            f.write(self.to_pgn() + "\n")
        return path
