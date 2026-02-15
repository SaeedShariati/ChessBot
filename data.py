# prepare_chess_data.py
import zstandard as zstd
import chess
import chess.pgn
import io
import torch
import numpy as np
from tqdm import tqdm
import random
import os
import glob
from multiprocessing import Pool, cpu_count

# ============================================
# PART 1: YOUR EXISTING FUNCTIONS
# ============================================

def safe_elo(elo_str):
    """Convert ELO safely"""
    try:
        if elo_str and elo_str != '?':
            return int(elo_str)
        return 0
    except:
        return 0

def board_to_tensor(board):
    """Convert a chess board to a 19x8x8 tensor"""
    # Piece channels (12: 6 piece types x 2 colors)
    # Plus: castling rights (4), en passant (1), turn (1), repetition (1)
    # Total: 12 + 4 + 1 + 1 + 1 = 19
    
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    
    # Piece positions
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - (square // 8)
            col = square % 8
            channel = piece_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
            tensor[channel, row, col] = 1.0
    
    # Castling rights (channels 12-15)
    if board.castling_rights & chess.BB_H1:  # White kingside
        tensor[12, 7, 4] = 1.0
    if board.castling_rights & chess.BB_A1:  # White queenside
        tensor[13, 7, 4] = 1.0
    if board.castling_rights & chess.BB_H8:  # Black kingside
        tensor[14, 0, 4] = 1.0
    if board.castling_rights & chess.BB_A8:  # Black queenside
        tensor[15, 0, 4] = 1.0
    
    # En passant (channel 16)
    if board.ep_square:
        row = 7 - (board.ep_square // 8)
        col = board.ep_square % 8
        tensor[16, row, col] = 1.0
    
    # Turn (channel 17) - 1 for white, 0 for black stored as 1 in channel 17 if white's turn
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0
    
    # Repetition count (channel 18) - simplified, just check if position repeated
    # For simplicity, we'll set this to 0 most of the time
    # In a real implementation, you'd track repetitions
    
    return torch.FloatTensor(tensor)

def enhanced_evaluate(board):
    """
    Position evaluation using multiple heuristics
    Returns value between -1 and 1
    """
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    
    score = 0
    
    # 1. Material (weight: 1.0)
    piece_values = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    # 2. Piece-square tables (positional bonuses) 
    # FIXED: Pawns on 8th rank are handled by material (they'll be queens!)
    pawn_table = [
        [0,  0,  0,  0,  0,  0,  0,  0],  # Rank 8 - promotion (bonus from material)
        [50, 50, 50, 50, 50, 50, 50, 50], # Rank 7 - near promotion
        [10, 10, 20, 30, 30, 20, 10, 10], # Rank 6
        [5,  5, 10, 25, 25, 10,  5,  5],  # Rank 5
        [0,  0,  0, 20, 20,  0,  0,  0],  # Rank 4
        [0,  0,  0,  0,  0,  0,  0,  0],  # Rank 3
        [0,  0,  0,  0,  0,  0,  0,  0],  # Rank 2
        [0,  0,  0,  0,  0,  0,  0,  0]   # Rank 1
    ]
    
    # Knights: center good, edges bad
    knight_table = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ]
    
    # Bishops: like center, avoid corners
    bishop_table = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ]
    
    # Rooks: prefer open files, 7th rank
    rook_table = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [0,  0,  0,  5,  5,  0,  0,  0]
    ]
    
    # Queens: combine rook and bishop patterns
    queen_table = [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [-5,  0,  5,  5,  5,  5,  0, -5],
        [0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ]
    
    # Apply piece-square tables
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
            
        row = square // 8
        col = square % 8
        
        # Flip row for black pieces
        if piece.color == chess.BLACK:
            row = 7 - row
        
        bonus = 0
        if piece.piece_type == chess.PAWN:
            # FIXED: Only give bonus if not on promotion rank
            if (piece.color == chess.WHITE and row < 7) or (piece.color == chess.BLACK and row > 0):
                bonus = pawn_table[row][col]
        elif piece.piece_type == chess.KNIGHT:
            bonus = knight_table[row][col]
        elif piece.piece_type == chess.BISHOP:
            bonus = bishop_table[row][col]
        elif piece.piece_type == chess.ROOK:
            bonus = rook_table[row][col]
        elif piece.piece_type == chess.QUEEN:
            bonus = queen_table[row][col]
        
        if piece.color == chess.WHITE:
            score += bonus
        else:
            score -= bonus
    
    # 3. Mobility (number of legal moves)
    white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    mobility_score = (white_moves - black_moves) * 2
    score += mobility_score
    
    # 4. Center control
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = 0
    for square in center_squares:
        if board.is_attacked_by(chess.WHITE, square):
            center_control += 5
        if board.is_attacked_by(chess.BLACK, square):
            center_control -= 5
    score += center_control
    
    # 5. King safety
    if board.has_castling_rights(chess.WHITE):
        score += 30
    if board.has_castling_rights(chess.BLACK):
        score -= 30
    
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    if white_king_square:
        king_file = chess.square_file(white_king_square)
        king_rank = chess.square_rank(white_king_square)
        if king_rank < 3:
            score -= 20
        for offset in [-1, 0, 1]:
            shield_square = chess.square(king_file + offset, king_rank + 1)
            if 0 <= shield_square < 64:
                piece = board.piece_at(shield_square)
                if not piece or piece.piece_type != chess.PAWN or piece.color != chess.WHITE:
                    score -= 10
    
    if black_king_square:
        king_file = chess.square_file(black_king_square)
        king_rank = chess.square_rank(black_king_square)
        if king_rank > 4:
            score += 20
        for offset in [-1, 0, 1]:
            shield_square = chess.square(king_file + offset, king_rank - 1)
            if 0 <= shield_square < 64:
                piece = board.piece_at(shield_square)
                if not piece or piece.piece_type != chess.PAWN or piece.color != chess.BLACK:
                    score += 10
    
    # Normalize to [-1, 1]
    return np.tanh(score / 1000.0)
def move_to_index(move):
    """Convert move to index (simplified but works)"""
    return move.from_square * 64 + move.to_square


def extract_positions_from_file(zst_path, min_elo=1500, max_positions=500000):
    """
    Extract positions and moves from a .zst file
    Returns: list of (board, next_move)
    """
    print(f"\nLoading {zst_path}...")
    
    positions_data = []  # Each element: (board, next_move)
    games_processed = 0
    games_kept = 0
    
    with open(zst_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            pbar = tqdm(desc="Processing games")
            
            while len(positions_data) < max_positions:
                try:
                    game = chess.pgn.read_game(text_stream)
                    if game is None:
                        break
                    
                    games_processed += 1
                    
                    # Check ELO
                    white_elo = safe_elo(game.headers.get('WhiteElo'))
                    black_elo = safe_elo(game.headers.get('BlackElo'))
                    
                    if white_elo >= min_elo and black_elo >= min_elo:
                        games_kept += 1
                        
                        board = game.board()
                        moves = list(game.mainline_moves())
                        
                        # For each position, record the next move
                        for i in range(len(moves) - 1):
                            # Position before move i
                            pos_board = game.board()
                            for move in moves[:i]:
                                pos_board.push(move)
                            
                            # The move played from this position
                            next_move = moves[i]
                            
                            positions_data.append((pos_board, next_move))
                            
                            if len(positions_data) >= max_positions:
                                break
                    
                    if games_processed % 1000 == 0:
                        pbar.set_description(f"Games: {games_kept}/{games_processed} | Positions: {len(positions_data)}")
                        
                except Exception as e:
                    continue
            
            pbar.close()
    
    print(f"\n✅ Extracted {len(positions_data)} positions from {games_kept} games")
    return positions_data

def prepare_training_data(positions_data):
    """
    Convert (board, next_move) pairs to tensors with evaluations
    """
    print("Converting to tensors and evaluating...")
    
    X = []
    y_value = []
    y_policy = []
    
    for board, next_move in tqdm(positions_data):
        X.append(board_to_tensor(board))
        y_value.append(enhanced_evaluate(board))
        y_policy.append(move_to_index(next_move))
    
    # Stack tensors
    X = torch.stack(X)
    y_value = torch.tensor(y_value, dtype=torch.float32)
    y_policy = torch.tensor(y_policy, dtype=torch.long)
    
    return X, y_value, y_policy

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    MIN_ELO = 1500
    MAX_POSITIONS = [500000,200000] # Total positions to collect
    
    files = [
        "lichess_db_standard_rated_2015-01.pgn.zst",
        "lichess_db_standard_rated_2013-01.pgn.zst",
    ]
    
    all_positions_data = []
    i = 0
    for file in files:
        if os.path.exists(file):
            positions_data = extract_positions_from_file(
                file, 
                min_elo=MIN_ELO,
                max_positions=MAX_POSITIONS[i]
            )
            all_positions_data.extend(positions_data)
        i = i+1
    
    print(f"\nTotal positions collected: {len(all_positions_data)}")
    
    # Convert to tensors
    X, y_value, y_policy = prepare_training_data(all_positions_data)
    
    # Save everything
    print("Saving training data...")
    torch.save({
        'X': X,
        'y_value': y_value,
        'y_policy': y_policy,
        'num_positions': len(X)
    }, 'chess_training_data.pt')
    
    print(f"\n✅ Done! Saved {len(X)} positions to chess_training_data.pt")
    print(f"   X shape: {X.shape}")
    print(f"   y_value shape: {y_value.shape}")
    print(f"   y_policy shape: {y_policy.shape}")