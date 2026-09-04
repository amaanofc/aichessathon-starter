"""The submission entrypoint. The platform imports this file and calls get_move."""

import random
import time

import chess

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

# --- Game-memory state (persists across calls within one game, per contract) ---
# A shadow board tracking the real game as played, kept in sync every call by
# inferring the opponent's move from the FEN we're handed. This gives the
# search real move history to work with (needed for board.is_repetition to
# mean anything), and it doubles as new-game detection: if we can't find a
# legal continuation that reproduces the incoming FEN, we treat it as fresh.
_game_board: "chess.Board | None" = None

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
RESERVE_MS = 450           # never touch this much of the clock, no matter what --
                           # bumped from 250 after observing a 286ms overshoot in
                           # real logs, which exceeded the old margin outright.
CRITICAL_MS = 550          # below this, don't search at all, just move -- kept
                           # comfortably above RESERVE_MS so the budget clamp
                           # (time_left - RESERVE_MS) can never go non-positive
                           # for any time_left_ms where we still decide to search.

# A fixed fraction of time_left decays geometrically move over move (spend 2%
# of what's left, 40 times, and ~45% of the original clock is still unspent)
# -- structurally guaranteed to leave most of the clock on the table. Divide
# by an estimate of remaining moves instead, so spending naturally rises as
# the game gets shorter. 30 is a rough starting guess for a full game length;
# tune it once real game-length data is available.
MOVES_LEFT_ESTIMATE = 30

# When the position we're being asked to move from has already occurred once
# this game, spend proportionally more of the budget looking for a way out
# instead of accepting the repeat -- this is the direct fix for contempt's
# real limitation: it can only steer away from a repeat if the search is deep
# enough to see an alternative, and the normal budget often isn't. Still
# passes through the exact same MAX_BUDGET_MS / RESERVE_MS ceilings below, so
# no matter how aggressive this multiplier is, it can never create new flag
# risk -- it only decides how to spend a budget that was already safe.
REPETITION_TIME_MULTIPLIER = 3.0

# How often (in node visits) negmax re-checks the wall clock. Kept low because
# per-node cost isn't free (move ordering, PST lookups); the check needs to
# fire often enough that the worst-case gap between checks stays small
# relative to the smallest budget we can ever have.
CHECK_INTERVAL = 256
assert CHECK_INTERVAL & (CHECK_INTERVAL - 1) == 0

# board.is_repetition() is a linear scan (bounded by halfmove_clock, but that
# can still be ~100 plies in a long shuffling endgame -- exactly the
# situation this exists to catch). Sampled rather than checked at every node,
# same idea as CHECK_INTERVAL, for the same reason.
REPETITION_CHECK_MASK = 63

# Only worth avoiding a repeat when clearly ahead; only worth actively
# steering toward one when it isn't. This is a contempt factor: a repeat
# scores worse than a plain draw for the side that's winning.
CONTEMPT_THRESHOLD = 150
CONTEMPT_PENALTY = 50

QUIESCENCE_MAX_PLY = 8

# --- Piece-square tables -----------------------------------------------------
# Standard "simplified evaluation function" values (Michniewski), White's
# perspective, a8 first. Indexed via square + square_mirror for Black, so
# only one table per piece is needed.
PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
ROOK_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]
QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
KING_MG_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]
KING_EG_PST = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]
PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}
# Non-pawn material at the start of the game, both sides combined. Used to
# taper the king table from middlegame to endgame as pieces come off.
PHASE_MAX = 2 * (2 * PIECE_VALUES[chess.KNIGHT] + 2 * PIECE_VALUES[chess.BISHOP]
                  + 2 * PIECE_VALUES[chess.ROOK] + PIECE_VALUES[chess.QUEEN])


class SearchTimeout(Exception):
    pass


def _log_move(depth, state, start_time, budget_ms, score, move, timed_out):
    # print() only -> stdout, flushed explicitly since a pipe is block-
    # buffered by default. No disk write here: a per-move open()/append()
    # to a fresh file handle is a real, if intermittent, source of stalls
    # on Windows (antivirus real-time scanning, disk contention), and a
    # handful of 100-300ms overshoots turned up in the last debug batch
    # that this synchronous file write is the leading suspect for. Not
    # worth the risk once print()-based logging has already proven it works.
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(
        f"[agent] move={move.uci()} depth={depth} nodes={state['nodes']} "
        f"elapsed_ms={elapsed_ms:.0f} budget_ms={budget_ms:.0f} "
        f"score={score} timed_out={timed_out} "
        f"repeats_seen={state['repeats_seen']} contempt_fired={state['contempt_fired']}",
        flush=True,
    )


def _sync_game_board(fen: str) -> None:
    """Update the persistent shadow board with whatever the opponent just
    played, inferred from the FEN we've been handed. Falls back to treating
    this as a new game if nothing matches (including the very first call)."""
    global _game_board
    incoming = chess.Board(fen)

    if _game_board is not None:
        target = incoming.epd()  # position only: no move counters
        for opp_move in _game_board.legal_moves:
            probe = _game_board.copy(stack=False)
            probe.push(opp_move)
            if probe.epd() == target:
                _game_board.push(opp_move)
                return
        # No legal continuation reproduces this position: new game.

    _game_board = incoming


def get_move(fen: str, time_left_ms: int) -> str:
    _sync_game_board(fen)
    board = _game_board.copy(stack=True)
    legal_move_list = list(board.legal_moves)

    if not legal_move_list:
        # The contract guarantees we're never asked to move in a terminal
        # position. If it ever happens anyway, there's no legal move to
        # return -- fail cleanly rather than crash on random.choice([]).
        return "0000"

    if len(legal_move_list) == 1:
        _game_board.push(legal_move_list[0])
        return legal_move_list[0].uci()

    panic_best = random.choice(legal_move_list)

    if time_left_ms <= CRITICAL_MS:
        _game_board.push(panic_best)
        return panic_best.uci()

    # If the position we're being asked to move from has already occurred
    # once this game, we're one repeat away from a forced draw -- worth
    # spending well beyond the normal per-move share to find a way out,
    # still bounded by the exact same hard ceilings as any other move.
    already_repeated = _game_board.is_repetition(2)
    base_share = time_left_ms / MOVES_LEFT_ESTIMATE
    if already_repeated:
        base_share *= REPETITION_TIME_MULTIPLIER

    # Budget is the minimum of three independent caps, so none of them can
    # individually push us past the real clock.
    time_budget_ms = min(
        MAX_BUDGET_MS,
        base_share,
        time_left_ms - RESERVE_MS,
    )
    time_budget_s = time_budget_ms / 1000.0

    start_time = time.perf_counter()
    deadline = start_time + time_budget_s
    search_state = {"nodes": 0, "deadline": deadline, "repeats_seen": 0, "contempt_fired": 0}

    best_move = None
    reached_depth = 0
    final_score = None

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
            chosen = best_move if best_move else panic_best
            _game_board.push(chosen)
            _log_move(reached_depth, search_state, start_time, time_budget_ms, final_score, chosen, timed_out=True)
            return chosen.uci()

        if depth_best_move is not None:
            best_move = depth_best_move
            reached_depth = depth
            final_score = depth_best_score

        if depth_best_score >= MATE_SCORE - MAX_DEPTH:
            break

    chosen = best_move if best_move else panic_best
    _game_board.push(chosen)
    _log_move(reached_depth, search_state, start_time, time_budget_ms, final_score, chosen, timed_out=False)
    return chosen.uci()


def negmax(board, depth, state, alpha, beta):
    state["nodes"] += 1
    if state["nodes"] & (CHECK_INTERVAL - 1) == 0:
        if time.perf_counter() > state["deadline"]:
            raise SearchTimeout

    # Repetition check: gated on halfmove_clock (a repeat is geometrically
    # impossible below 4, since it takes at least two round trips) and
    # sampled rather than checked every node, since the scan has real cost.
    if (
        board.halfmove_clock >= 4
        and state["nodes"] & REPETITION_CHECK_MASK == 0
        and board.is_repetition(2)
    ):
        state["repeats_seen"] += 1
        mover_edge = evaluate(board)
        if mover_edge >= CONTEMPT_THRESHOLD:
            state["contempt_fired"] += 1
            return -CONTEMPT_PENALTY
        return 0

    if depth == 0:
        return quiescence(board, state, alpha, beta)

    moves = order_moves(board)
    if not moves:
        # No legal moves: checkmate or stalemate. Reusing the move list we
        # already generated avoids a second board.is_game_over() traversal.
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


def quiescence(board, state, alpha, beta, qdepth=0):
    """Keep resolving captures (and check-evasions) past the search horizon
    so a static eval never gets caught mid-exchange. Hard-capped in ply so a
    long forcing sequence can't itself become a timing risk."""
    state["nodes"] += 1
    if state["nodes"] & (CHECK_INTERVAL - 1) == 0:
        if time.perf_counter() > state["deadline"]:
            raise SearchTimeout

    in_check = board.is_check()

    if not in_check:
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        if qdepth >= QUIESCENCE_MAX_PLY:
            return alpha
        moves = order_captures(board)
        if not moves:
            return alpha
    else:
        # Can't stand pat while in check -- must consider every legal reply.
        moves = order_moves(board)
        if not moves:
            return -MATE_SCORE

    for move in moves:
        board.push(move)
        try:
            score = -quiescence(board, state, -beta, -alpha, qdepth + 1)
        finally:
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def evaluate(board):
    score = 0
    non_pawn_material = 0
    white_king_sq = None
    black_king_sq = None

    for square, piece in board.piece_map().items():
        pt = piece.piece_type
        if pt == chess.KING:
            if piece.color == chess.WHITE:
                white_king_sq = square
            else:
                black_king_sq = square
            continue

        value = PIECE_VALUES[pt]
        idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
        pst_val = PST[pt][idx]
        if piece.color == chess.WHITE:
            score += value + pst_val
        else:
            score -= value + pst_val
        if pt != chess.PAWN:
            non_pawn_material += value

    phase = min(1.0, non_pawn_material / PHASE_MAX)  # 1.0 = full material, 0.0 = bare endgame

    if white_king_sq is not None:
        score += KING_MG_PST[white_king_sq] * phase + KING_EG_PST[white_king_sq] * (1 - phase)
    if black_king_sq is not None:
        idx = chess.square_mirror(black_king_sq)
        score -= KING_MG_PST[idx] * phase + KING_EG_PST[idx] * (1 - phase)

    return score if board.turn == chess.WHITE else -score


def _capture_value(board, move):
    if board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    return PIECE_VALUES.get(victim, 0)


def order_moves(board, prev_best=None):
    moves = list(board.legal_moves)

    def score(move):
        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square)
            return 100000 + (10 * _capture_value(board, move)) - PIECE_VALUES.get(attacker, 0)
        return 0

    moves.sort(key=score, reverse=True)

    if prev_best and prev_best in moves:
        moves.remove(prev_best)
        moves.insert(0, prev_best)

    return moves


def order_captures(board):
    moves = list(board.generate_legal_captures())

    def score(move):
        attacker = board.piece_type_at(move.from_square)
        return (10 * _capture_value(board, move)) - PIECE_VALUES.get(attacker, 0)

    moves.sort(key=score, reverse=True)
    return moves