import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
import zipfile
import io
import os
from collections import deque
import math
import random
from tqdm import tqdm

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        fc_dropout=0.3
        self.features = nn.Sequential(
            nn.Conv2d(19, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
        )
        
        # Policy 
        self.policy = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 4096)
        )
        
        # Value 
        self.value = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.policy(features), self.value(features)

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
    
    # Repetition count, ignore for now
    
    return torch.FloatTensor(tensor)
def tensor_to_board(tensor):
    """Convert a 19x8x8 tensor back to a chess board (for validation)"""
    board = chess.Board()
    board.clear()  # Start with empty board
    
    tensor = tensor.numpy() if torch.is_tensor(tensor) else tensor
    
    # Piece mapping (reverse of board_to_tensor)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                   chess.ROOK, chess.QUEEN, chess.KING]
    
    # White pieces (channels 0-5)
    for channel, piece_type in enumerate(piece_types):
        for row in range(8):
            for col in range(8):
                if tensor[channel, row, col] > 0.5:
                    square = chess.square(col, 7-row)  # Convert row back
                    board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
    
    # Black pieces (channels 6-11)
    for channel, piece_type in enumerate(piece_types):
        for row in range(8):
            for col in range(8):
                if tensor[channel+6, row, col] > 0.5:
                    square = chess.square(col, 7-row)
                    board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))
    
    # Set turn (channel 17)
    if tensor[17, 0, 0] > 0.5:
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK
    
    # Note: castling rights, en passant are lost in conversion
    # But for validation, material evaluation should still work
    
    return board
import io
import random


def move_to_index(move):
    """Convert a move to an index (simplified - real would need more careful mapping)"""
    return move.from_square * 64 + move.to_square

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

class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.visits = 0
        self.value_sum = 0
        self.children = {}
        self.is_expanded = False
    
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def ucb_score(self, exploration_constant=1.2):
        if self.visits == 0:
            return float('inf')
        
        # PUCT algorithm
        exploration = exploration_constant * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploration + self.value()


class MCTS:
    def __init__(self, model, num_simulations=800):
        self.model = model
        self.num_simulations = num_simulations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def search(self, board):
        root = MCTSNode(board.copy())
        
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            
            # Selection
            while node.is_expanded and node.children:
                # Select best child using UCB
                best_score = -float('inf')
                best_child = None
                
                for child in node.children.values():
                    score = child.ucb_score()
                    if score > best_score:
                        best_score = score
                        best_child = child
                
                node = best_child
                path.append(node)
            
            # Expansion
            if not node.board.is_game_over() and not node.is_expanded:
                self.expand(node)
                node.is_expanded = True
            
            # Evaluation
            if node.board.is_game_over():
                # Game over
                result = node.board.result()
                if result == "1-0":
                    value = 1.0
                elif result == "0-1":
                    value = -1.0
                else:
                    value = 0.0
            else:
                # Use neural network for evaluation
                value = self.evaluate(node.board)
            
            # Backpropagation
            for node in reversed(path):
                node.visits += 1
                # Adjust value based on whose turn it was
                if node.board.turn == board.turn:
                    node.value_sum += value
                else:
                    node.value_sum -= value
        
        return root
    
    def expand(self, node):
        board = node.board
        
        # Get policy from neural network
        policy, _ = self.predict(board)
        
        for move in board.legal_moves:
            new_board = board.copy()
            new_board.push(move)
            
            # Get prior probability for this move
            move_idx = move_to_index(move)
            prior = F.softmax(policy, dim=1)[0, move_idx].item()
            
            child = MCTSNode(new_board, parent=node, move=move, prior=prior)
            node.children[move] = child
    
    def predict(self, board):
        with torch.no_grad():
            tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            policy, value = self.model(tensor)
            return policy, value
    
    def evaluate(self, board):
        _, value = self.predict(board)
        return value.item()
    
    def get_best_move(self, root):
        # Choose the most visited child
        best_child = max(root.children.items(), key=lambda item: item[1].visits)
        return best_child[0]









def validate_training_data(data_file="./chess_data/chess_data_batch_0000.pt", num_samples=100):
    """
    Compare stored training values with fresh heuristic evaluations
    """
    print(f"\n🔍 VALIDATING TRAINING DATA: {data_file}")
    print("="*60)
    
    # Load training data
    data = torch.load(data_file)
    X = data['X']
    y_value = data['y_value']
    y_policy = data['y_policy']
    
    print(f"Total positions: {len(X)}")
    print(f"Sampling {num_samples} random positions...\n")
    
    # Sample random indices
    indices = random.sample(range(len(X)), min(num_samples, len(X)))
    
    discrepancies = []
    
    for idx in indices:
        # Get stored value
        stored_value = y_value[idx].item()
        
        # We need to reconstruct the board from tensor
        # This is tricky - we need a tensor_to_board function
        # For now, let's assume you have one or we'll skip this part
        
        print(f"Position {idx}:")
        print(f"  Stored value: {stored_value:.4f}")
        # print(f"  Fresh heuristic: {fresh_value:.4f}")
        # print(f"  Difference: {abs(stored_value - fresh_value):.4f}")
        print()
    
    # If we could compare, we'd print statistics
    # avg_diff = sum(discrepancies) / len(discrepancies)
    # print(f"\nAverage difference: {avg_diff:.4f}")
    # print(f"Max difference: {max(discrepancies):.4f}")

def validate_training_data_full(data_file="./chess_data/chess_data_batch_0000.pt", num_samples=100):
    """
    Compare stored training values with fresh heuristic evaluations
    """
    print(f"\n🔍 VALIDATING TRAINING DATA: {data_file}")
    print("="*60)
    
    # Load training data
    data = torch.load(data_file)
    X = data['X']
    y_value = data['y_value']
    
    print(f"Total positions: {len(X)}")
    print(f"Sampling {num_samples} random positions...\n")
    
    indices = random.sample(range(len(X)), min(num_samples, len(X)))
    
    stored_values = []
    fresh_values = []
    differences = []
    
    for idx in indices:
        # Get stored value
        stored = y_value[idx].item()
        stored_values.append(stored)
        
        # Reconstruct board
        board = tensor_to_board(X[idx])
        
        # Get fresh heuristic evaluation
        fresh = enhanced_evaluate(board)
        fresh_values.append(fresh)
        
        diff = abs(stored - fresh)
        differences.append(diff)
        
        print(f"Position {idx}:")
        print(f"  Stored: {stored:.4f}, Fresh: {fresh:.4f}, Diff: {diff:.4f}")
        if diff > 0.2:
            print(f"  ⚠️ Large discrepancy!")
        print()
    
    # Statistics
    avg_diff = sum(differences) / len(differences)
    max_diff = max(differences)
    
    print("="*60)
    print(f"SUMMARY:")
    print(f"  Average difference: {avg_diff:.4f}")
    print(f"  Max difference: {max_diff:.4f}")
    print(f"  Stored value range: [{min(stored_values):.4f}, {max(stored_values):.4f}]")
    print(f"  Fresh value range: [{min(fresh_values):.4f}, {max(fresh_values):.4f}]")
    
    if avg_diff < 0.1:
        print("\n✅ Training data is consistent with heuristic!")
    elif avg_diff < 0.2:
        print("\n⚠️ Training data has some noise but should be OK")
    else:
        print("\n❌ Training data is corrupted! Values don't match heuristic")
    
    return avg_diff, max_diff
class ChessEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print("Initializing new model")
        
        self.mcts = None
    def train_from_batches(self, data_prefix="chess_data", epochs=10, batch_size=64):
        """
        Train using batched data files and save models with validation loss in filename
        """
        import glob
        
        print(f"Loading batches from ./chess_data/{data_prefix}_batch_*.pt...")
        
        # Find all batch files
        batch_files = sorted(glob.glob(f"./chess_data/{data_prefix}_batch_*.pt"))
        
        if not batch_files:
            print("No batch files found!")
            return
        
        # Split into train/val (80/20 by files)
        split = int(0.8 * len(batch_files))
        train_files = batch_files[:split]
        val_files = batch_files[split:]
        
        print(f"Found {len(batch_files)} batches: {len(train_files)} train, {len(val_files)} val")
        
        # Simple dataset class
        class SimpleChessDataset(torch.utils.data.IterableDataset):
            def __init__(self, files):
                self.files = files
            
            def __iter__(self):
                for f in self.files:
                    data = torch.load(f)
                    for i in range(len(data['X'])):
                        yield data['X'][i], data['y_value'][i], data['y_policy'][i]
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            SimpleChessDataset(train_files),
            batch_size=batch_size,
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            SimpleChessDataset(val_files),
            batch_size=batch_size,
            num_workers=0
        )
        
        # Training setup
        device = self.device
        self.model = self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        print(f"\nStarting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_X, batch_v, batch_p in train_loader:
                batch_X = batch_X.to(device)
                batch_v = batch_v.to(device).unsqueeze(1)
                batch_p = batch_p.to(device)
                
                optimizer.zero_grad()
                policy_out, value_out = self.model(batch_X)
                
                loss = 10*(F.mse_loss(value_out, batch_v) + 
                    F.cross_entropy(policy_out, batch_p))
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_v, batch_p in val_loader:
                    batch_X = batch_X.to(device)
                    batch_v = batch_v.to(device).unsqueeze(1)
                    batch_p = batch_p.to(device)
                    
                    policy_out, value_out = self.model(batch_X)
                    loss = (F.mse_loss(value_out, batch_v) + 
                        F.cross_entropy(policy_out, batch_p))
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_train = train_loss / train_batches
            avg_val = val_loss / val_batches
            
            # Track best model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_filename = f"chess_model_BEST_val_{best_val_loss:.4f}.pth"
                # Save a copy as the best model
                torch.save(self.model.state_dict(), best_filename)
                print(f"Epoch {epoch+1}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f} ✓ BEST SO FAR")
            else:
                print(f"Epoch {epoch+1}: Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best model saved as: chess_model_BEST_val_{best_val_loss:.4f}.pth")
    # Keep your existing play_move and play_game methods
    def play_move(self, board, num_simulations=800):
        if self.mcts is None:
            self.mcts = MCTS(self.model, num_simulations)
        root = self.mcts.search(board)
        return self.mcts.get_best_move(root)
    
    def play_game(self, opponent=None, num_moves=100):
        board = chess.Board()
        for move_num in range(num_moves):
            print(f"\nMove {move_num + 1}")
            print(board)
            
            if board.is_game_over():
                print(f"Game over: {board.result()}")
                break
            
            if board.turn == chess.WHITE:
                move = self.play_move(board)
                print(f"Engine plays: {move}")
            else:
                if opponent == "random":
                    move = random.choice(list(board.legal_moves))
                    print(f"Random plays: {move}")
                elif opponent == "human":
                    print("Enter your move (e.g., e2e4):")
                    move_uci = input()
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move not in board.legal_moves:
                            print("Illegal move, try again")
                            continue
                    except:
                        print("Invalid format, try again")
                        continue
                else:
                    move = self.play_move(board)
                    print(f"Engine (black) plays: {move}")
            
            board.push(move)
        
        return board
    def debug_position(self, fen=None):
        """Debug what the network sees in various positions"""
        
        test_positions = [
            ("Hanging Queen", "rnb1kbnr/pppp1ppp/8/4q3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", 
            "White can win queen", 0.5),
            
            ("Simple Checkmate in 1", "4k3/8/8/8/8/8/8/4K2Q w - - 0 1",
            "White to play Qe8#", 1.0),
            
            ("Back Rank Mate in 1", "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
            "White to play Rf8#", 1.0),
            
            ("Equal Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Starting position", 0.0),
        ]
        
        print(f"\n{'='*60}")
        print(f"DEBUGGING MULTIPLE POSITIONS")
        print(f"{'='*60}")
        
        for desc, test_fen, explanation, expected in test_positions:
            board = chess.Board(test_fen)
            print(f"\n{desc}: {explanation}")
            print(f"FEN: {test_fen}")
            print(board)
            
            # Heuristic evaluation
            heur_val = enhanced_evaluate(board)
            print(f"📊 Heuristic: {heur_val:.4f} (Expected: ~{expected})")
            
            # Network evaluation
            self.model.eval()
            with torch.no_grad():
                tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
                policy, net_val = self.model(tensor)
            
            print(f"🧠 Network: {net_val.item():.4f}")
            
            # Top moves
            probs = torch.softmax(policy[0], dim=0)
            top_probs, top_indices = torch.topk(probs, 10)
            
            print(f"\n🎯 Top 5 moves:")
            for i in range(5):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                from_sq = idx // 64
                to_sq = idx % 64
                move = chess.Move(from_sq, to_sq)
                if move in board.legal_moves:
                    print(f"  {board.san(move)}: {prob:.4f}")
            
            print("-" * 40)
# ============================================
# PART 6: DEMO AND TESTING
# ============================================

def main():
    print("=" * 50)
    print("PyTorch Chess Engine with MCTS")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU found")
    
    engine = ChessEngine()
    
    while True:
        print("\nOptions:")
        print("1. Train model")
        print("2. Load existing model")
        print("3. Play against engine")
        print("4. Watch engine vs random")
        print("5. Watch engine self-play")
        print("6. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            epochs = int(input("Number of epochs (5-20): "))
            batch_size = 64
            
            engine.train_from_batches(
                data_prefix="chess_data",  # This matches your batch files: chess_data_batch_*.pt
                epochs=epochs,
                batch_size=batch_size
            )
        
        elif choice == "2":
            model_path = input("Enter model path (default: chess_model.pth): ") or "chess_model.pth"
            if os.path.exists(model_path):
                engine.model.load_state_dict(torch.load(model_path, map_location=engine.device))
                print("Model loaded")
            else:
                print("Model not found")
        
        elif choice == "3":
            print("\nPlaying against engine (you are black)")
            print("Enter moves in UCI format (e.g., e2e4)")
            engine.play_game(opponent="human")
        
        elif choice == "4":
            print("\nWatching engine vs random moves")
            engine.play_game(opponent="random")
        
        elif choice == "5":
            print("\nWatching engine play against itself")
            engine.play_game(opponent="self")
        
        elif choice == "6":
            break
        elif choice == "7":
            #debug
            engine.debug_position()
            validate_training_data_full()


if __name__ == "__main__":
    main()



