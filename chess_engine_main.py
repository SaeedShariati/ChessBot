# chess_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
import requests
import zipfile
import io
import os
from collections import deque
import math
import random
from tqdm import tqdm

# ============================================
# PART 1: NEURAL NETWORK ARCHITECTURE
# ============================================

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Common feature extraction layers using Sequential
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(19, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 4096)
        )
        
        # Value head
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
    
    # Repetition count (channel 18) - simplified, just check if position repeated
    # For simplicity, we'll set this to 0 most of the time
    # In a real implementation, you'd track repetitions
    
    return torch.FloatTensor(tensor)


# ============================================
# PART 2: DATA COLLECTION
# ============================================
from datasets import load_dataset
import io
import random

def download_lichess_dataset(num_games=100, min_elo=2000, max_positions=10000):
    """
    Download real Lichess games from strong players (2000+ ELO)
    
    Args:
        num_games: Number of games to process
        min_elo: Minimum ELO rating for both players
        max_positions: Maximum positions to return (to avoid memory issues)
    
    Returns:
        List of chess.Board positions
    """
    print(f"Loading real Lichess games (ELO > {min_elo}) from Hugging Face...")
    
    positions = []
    games_found = 0
    total_positions = 0
    
    # Load June 2013 games in streaming mode
    try:
        dataset = load_dataset("Icannos/lichess_games", "2013-06", streaming=True, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to synthetic positions for testing...")
        return generate_synthetic_positions(num_games)
    
    for game_data in dataset:
        if games_found >= num_games or total_positions >= max_positions:
            break
            
        try:
            # Parse the game
            game = chess.pgn.read_game(io.StringIO(game_data['text']))
            if game is None:
                continue
            
            # Get player ratings
            if( game.headers.get('WhiteElo', 0) == '?' or game.headers.get('BlackElo', 0) == '?'):
                print(f"Game skipped, no elo.")
                continue
            white_elo = int(game.headers.get('WhiteElo', 0))
            black_elo = int(game.headers.get('BlackElo', 0))
            
            # Only keep high-quality games
            if white_elo >= min_elo and black_elo >= min_elo:
                board = game.board()
                game_positions = 0
                
                # Extract positions from the game
                for move in game.mainline_moves():
                    board.push(move)
                    
                    # Sample 30% of positions to keep dataset diverse
                    if random.random() < 0.3:
                        positions.append(board.copy())
                        total_positions += 1
                        game_positions += 1
                        
                        if total_positions >= max_positions:
                            break
                
                games_found += 1
                print(f"Game {games_found}: {white_elo} vs {black_elo} - Added {game_positions} positions")
                
        except Exception as e:
            print(f"Error processing game: {e}")
            continue
    
    print(f"\nLoaded {len(positions)} positions from {games_found} strong games")
    print(f"Average positions per game: {len(positions)/games_found:.1f}")
    
    return positions


#def download_lichess_dataset(num_games=100):
#    """Download a small dataset of Lichess games for training"""
#    print("Downloading Lichess games...")
#    
#    # Using the Lichess open database (small sample for demonstration)
##    # In practice, you'd want to download a larger dataset
#    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
#    
#    # For this example, we'll generate synthetic positions instead
#    # This avoids the large download
#    print("Using synthetic positions for demonstration")
#    print("To use real games, download from https://database.lichess.org/")
##   positions = []
 #   for _ in range(num_games * 10):  # Generate 10 positions per game
 #       board = chess.Board()
 #       # Play random moves to generate positions
 #       for _ in range(random.randint(10, 40)):
 #           if board.is_game_over():
 #               break
 #           move = random.choice(list(board.legal_moves))
 #           board.push(move)
 #       positions.append(board.copy())
 #   return positions


def generate_training_data(positions, stockfish_path=None):
    """Generate training data from positions using Stockfish evaluation"""
    print("Generating training data...")
    
    X = []
    y_value = []
    y_policy = []
    
    # If Stockfish is not available, use a heuristic evaluation
    for board in tqdm(positions):
        # Convert board to tensor
        X.append(board_to_tensor(board))
        
        # Generate value target (simple heuristic if no Stockfish)
        #value = simple_evaluate(board)
        value = enhanced_evaluate(board)
        y_value.append(value)
        
        # Generate policy target (random for now - you'd use actual moves in real training)
        if board.legal_moves:
            move = random.choice(list(board.legal_moves))
            policy_target = move_to_index(move)
        else:
            policy_target = 0
        y_policy.append(policy_target)
    
    return torch.stack(X), torch.tensor(y_value), torch.tensor(y_policy)

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
    
    # 2. Piece-square tables (positional bonuses) (weight: 0.3)
    # Pawns: advance is good
    pawn_table = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
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
        
        # Flip row for black pieces (they see board from other side)
        if piece.color == chess.BLACK:
            row = 7 - row
        
        bonus = 0
        if piece.piece_type == chess.PAWN:
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
    
    # 3. Mobility (number of legal moves) (weight: 0.1)
    white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
    black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    mobility_score = (white_moves - black_moves) * 2
    score += mobility_score
    
    # 4. Center control (pieces attacking center squares) (weight: 0.2)
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    center_control = 0
    for square in center_squares:
        if board.is_attacked_by(chess.WHITE, square):
            center_control += 5
        if board.is_attacked_by(chess.BLACK, square):
            center_control -= 5
    score += center_control
    
    # 5. King safety (weight: 0.2)
    if board.has_castling_rights(chess.WHITE):
        score += 30  # Castling rights are valuable
    if board.has_castling_rights(chess.BLACK):
        score -= 30
    
    # Penalize exposed kings
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    if white_king_square:
        # Pawn shield in front of king
        king_file = chess.square_file(white_king_square)
        king_rank = chess.square_rank(white_king_square)
        if king_rank < 3:  # King not castled yet
            score -= 20
        # Check if pawns are in front
        for offset in [-1, 0, 1]:
            shield_square = chess.square(king_file + offset, king_rank + 1)
            if 0 <= shield_square < 64:
                piece = board.piece_at(shield_square)
                if not piece or piece.piece_type != chess.PAWN or piece.color != chess.WHITE:
                    score -= 10
    
    if black_king_square:
        king_file = chess.square_file(black_king_square)
        king_rank = chess.square_rank(black_king_square)
        if king_rank > 4:  # Black king not castled
            score += 20
        for offset in [-1, 0, 1]:
            shield_square = chess.square(king_file + offset, king_rank - 1)
            if 0 <= shield_square < 64:
                piece = board.piece_at(shield_square)
                if not piece or piece.piece_type != chess.PAWN or piece.color != chess.BLACK:
                    score += 10
    
    # Normalize to [-1, 1]
    # 2000 centipawns = 20 pawn advantage = near 1.0
    return np.tanh(score / 2000.0)
def simple_evaluate(board):
    """Simple material-based evaluation (fallback if no Stockfish)"""
    if board.is_checkmate():
        return -1.0 if board.turn == chess.WHITE else 1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    # Normalize to [-1, 1]
    return np.tanh(score / 20.0)


def move_to_index(move):
    """Convert a move to an index (simplified - real would need more careful mapping)"""
    return move.from_square * 64 + move.to_square


# ============================================
# PART 3: TRAINING
# ============================================

def train_model(model, X, y_value, y_policy, epochs=10, batch_size=64):
    """Train the neural network"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    dataset = torch.utils.data.TensorDataset(X, y_value, y_policy)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_X, batch_y_value, batch_y_policy in dataloader:
            batch_X = batch_X.to(device)
            batch_y_value = batch_y_value.to(device).float().unsqueeze(1)
            batch_y_policy = batch_y_policy.to(device)
            
            optimizer.zero_grad()
            
            policy_out, value_out = model(batch_X)
            
            # Losses
            value_loss = F.mse_loss(value_out, batch_y_value)
            policy_loss = F.cross_entropy(policy_out, batch_y_policy)
            
            loss = value_loss + policy_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


# ============================================
# PART 4: MCTS IMPLEMENTATION
# ============================================

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
    
    def ucb_score(self, exploration_constant=1.4):
        if self.visits == 0:
            return float('inf')
        
        # PUCT algorithm
        exploration = exploration_constant * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + exploration


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
                # Game over - get actual outcome
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
        """Expand a node by adding all legal moves as children"""
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
        """Get neural network predictions for a board"""
        with torch.no_grad():
            tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            policy, value = self.model(tensor)
            return policy, value
    
    def evaluate(self, board):
        """Evaluate a board position using the neural network"""
        _, value = self.predict(board)
        return value.item()
    
    def get_best_move(self, root):
        """Select the best move from the MCTS root"""
        # Choose the most visited child
        best_child = max(root.children.items(), key=lambda item: item[1].visits)
        return best_child[0]


# ============================================
# PART 5: MAIN ENGINE CLASS
# ============================================

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
    
    def train(self, num_games=100, epochs=10):
        """Train the engine on Lichess games"""
        print("Starting training...")
        
        # Download games
        positions = download_lichess_dataset(num_games)
        
        # Generate training data
        X, y_value, y_policy = generate_training_data(positions)
        
        # Train model
        self.model = train_model(self.model, X, y_value, y_policy, epochs=epochs)
        
        # Save model
        torch.save(self.model.state_dict(), "chess_model.pth")
        print("Model saved to chess_model.pth")
    
    def play_move(self, board, num_simulations=800):
        """Choose a move using MCTS"""
        if self.mcts is None:
            self.mcts = MCTS(self.model, num_simulations)
        
        root = self.mcts.search(board)
        return self.mcts.get_best_move(root)
    
    def play_game(self, opponent=None, num_moves=100):
        """Play a game against an opponent (or random moves)"""
        board = chess.Board()
        
        for move_num in range(num_moves):
            print(f"\nMove {move_num + 1}")
            print(board)
            
            if board.is_game_over():
                print(f"Game over: {board.result()}")
                break
            
            if board.turn == chess.WHITE:
                # Our engine plays white
                move = self.play_move(board)
                print(f"Engine plays: {move}")
            else:
                # Opponent plays black
                if opponent == "random":
                    move = random.choice(list(board.legal_moves))
                    print(f"Random plays: {move}")
                else:
                    # Human input
                    print("Enter your move (e.g., e2e4):")
                    move_uci = input()
                    move = chess.Move.from_uci(move_uci)
            
            board.push(move)
        
        return board


# ============================================
# PART 6: DEMO AND TESTING
# ============================================

def main():
    print("=" * 50)
    print("PyTorch Chess Engine with MCTS")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU found - training will be slow")
    
    # Initialize engine
    engine = ChessEngine()
    
    # Menu
    while True:
        print("\nOptions:")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Play against engine")
        print("4. Watch engine vs random")
        print("5. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            num_games = int(input("Number of games to train on (100-1000): "))
            epochs = int(input("Number of epochs (5-20): "))
            engine.train(num_games=num_games, epochs=epochs)
        
        elif choice == "2":
            model_path = input("Enter model path (default: chess_model.pth): ") or "chess_model.pth"
            if os.path.exists(model_path):
                engine.model.load_state_dict(torch.load(model_path))
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
            break


if __name__ == "__main__":
    main()