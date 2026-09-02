"""The submission entrypoint. The platform imports this file and calls get_move."""

import random
import time

import chess

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

MATE_SCORE = 99999
MAX_DEPTH = 18

# --- Timing constants -------------------------------------------------------
# The platform enforces the clock on wall time, independent of anything we
# compute internally. These constants exist to guarantee, by construction,
# that our internal budget can never exceed the real clock, and that the
# gap between "we should stop" and "we actually notice" stays small.
MAX_BUDGET_MS = 5000       # never plan to think longer than this
BUDGET_FRACTION = 0.02     # ... or this fraction of the clock, whichever is smaller
RESERVE_MS = 250           # never touch this much of the clock, no matter what
CRITICAL_MS = 300          # below this, don't search at all, just move

# How often (in node visits) negmax re-checks the wall clock. Lower = safer,
# more time.time() overhead. This was 2048; it's lower now because per-node
# cost went up when move ordering was added, and the old interval was tuned
# for a cheaper per-node cost.
CHECK_INTERVAL = 256
assert CHECK_INTERVAL & (CHECK_INTERVAL - 1) == 0  # must be a power of two for the bitmask


class SearchTimeout(Exception):
    pass


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal_move_list = list(board.legal_moves)

    if not legal_move_list:
        # The contract guarantees we're never asked to move in a terminal
        # position. If it ever happens anyway, there is no legal move to
        # return -- fail cleanly rather than crash on random.choice([]).
        return "0000"

    if len(legal_move_list) == 1:
        # No decision to make. Don't spend any clock or CPU searching it.
        return legal_move_list[0].uci()

    panic_best = random.choice(legal_move_list)

    if time_left_ms <= CRITICAL_MS:
        # Too little time left to safely run even one checked iteration.
        # Move immediately rather than risk any overshoot at all.
        return panic_best.uci()

    # Budget is the minimum of three independent caps, so none of them can
    # individually push us past the real clock:
    #   - an absolute cap (MAX_BUDGET_MS)
    #   - a fraction of the clock (BUDGET_FRACTION)
    #   - the clock itself, minus a reserve we never spend
    time_budget_ms = min(
        MAX_BUDGET_MS,
        time_left_ms * BUDGET_FRACTION,
        time_left_ms - RESERVE_MS,
    )
    time_budget_s = time_budget_ms / 1000.0

    start_time = time.perf_counter()
    deadline = start_time + time_budget_s
    search_state = {"nodes": 0, "deadline": deadline}

    best_move = None

    for depth in range(1, MAX_DEPTH):
        alpha = -float("inf")
        beta = float("inf")

        depth_best_move = None
        depth_best_score = -float("inf")

        try:
            for move in order_moves(board, best_move):
                if time.perf_counter() > search_state["deadline"]:
                    raise SearchTimeout

                board.push(move)
                try:
                    score = -negmax(board, depth - 1, search_state, -beta, -alpha)
                finally:
                    board.pop()

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
                    alpha = max(alpha, depth_best_score)

        except SearchTimeout:
            return best_move.uci() if best_move else panic_best.uci()

        if depth_best_move is not None:
            best_move = depth_best_move

        # Found a forced mate; no point searching deeper.
        if depth_best_score >= MATE_SCORE - MAX_DEPTH:
            break

    return best_move.uci() if best_move else panic_best.uci()


def negmax(board, depth, state, alpha, beta):
    state["nodes"] += 1
    if state["nodes"] & (CHECK_INTERVAL - 1) == 0:
        if time.perf_counter() > state["deadline"]:
            raise SearchTimeout

    if depth == 0:
        return evaluate(board)

    moves = order_moves(board)
    if not moves:
        # No legal moves: checkmate or stalemate. Reusing the move list we
        # already generated avoids a second, separate board.is_game_over()
        # traversal (which also checks draw-by-repetition/50-move state
        # that we don't need mid-search) at every internal node.
        return -MATE_SCORE if board.is_check() else 0

    max_score = -float("inf")
    for move in moves:
        board.push(move)
        try:
            score = -negmax(board, depth - 1, state, -beta, -alpha)
        finally:
            board.pop()
        if score > max_score:
            max_score = score
            alpha = max(alpha, max_score)
        if alpha >= beta:
            break
    return max_score


def evaluate(board):
    score = 0
    for piece in PIECE_VALUES:
        score += len(board.pieces(piece, chess.WHITE)) * PIECE_VALUES[piece]
        score -= len(board.pieces(piece, chess.BLACK)) * PIECE_VALUES[piece]
    return score if board.turn == chess.WHITE else -score


def order_moves(board, prev_best=None):
    moves = list(board.legal_moves)

    def score(move):
        if board.is_capture(move):
            # piece_type_at returns a plain int and skips building a Piece
            # object, unlike piece_at -- cheaper, called on every move here.
            victim = board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            victim_value = PIECE_VALUES.get(victim, 0)
            attacker_value = PIECE_VALUES.get(attacker, 0)
            return 100000 + (10 * victim_value) - attacker_value
        return 0

    moves.sort(key=score, reverse=True)

    if prev_best and prev_best in moves:
        moves.remove(prev_best)
        moves.insert(0, prev_best)

    return moves