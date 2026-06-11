"""Tests for board_state.py - run with: python3 test_board_state.py

No camera or model needed: board readouts are simulated from python-chess
positions, so this verifies the FEN conversion and move tracking logic only.
"""

import chess

from board_state import predictions_to_placement, GameTracker


def board_to_preds(board):
    """Builds our 64-label readout list from a chess.Board (row 0 = rank 8)."""
    preds = []
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(chess.square(c, 7 - r))
            if piece is None:
                preds.append("empty")
            else:
                color = "w" if piece.color else "b"
                preds.append(color + piece.symbol().lower())
    return preds


def main():
    # 1. Starting position converts to the correct FEN placement
    start = chess.Board()
    placement = predictions_to_placement(board_to_preds(start))
    assert placement == start.board_fen(), placement
    print(f"1. Start position FEN OK: {placement}")

    # 2. A full move sequence is detected, including a capture and castling
    tracker = GameTracker()
    game = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f6e4"]:
        game.push_uci(uci)
        result = tracker.update(board_to_preds(game))
        assert result["status"] == "move", result
        print(f"2. Detected move: {result['san']}")
    assert tracker.fen == game.fen(), (tracker.fen, game.fen())
    print("2. Tracker FEN matches game (incl. turn + castling rights) OK")

    # 3. Feeding the same readout again reports no change
    assert tracker.update(board_to_preds(game))["status"] == "no_change"
    print("3. Duplicate readout -> no_change OK")

    # 4. An impossible readout (two white kings) is rejected, state untouched
    bad = board_to_preds(game)
    bad[20] = "wk"
    result = tracker.update(bad)
    assert result["status"] == "unrecognized", result
    assert tracker.fen == game.fen()
    print("4. Noisy readout rejected, game state untouched OK")

    # 5. Board back at the start position is detected as a new game
    result = tracker.update(board_to_preds(chess.Board()))
    assert result["status"] == "new_game" and tracker.moves == []
    print("5. Reset to start position -> new_game OK")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
