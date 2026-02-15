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

import io
import random


def move_to_index(move):
    """Convert a move to an index (simplified - real would need more careful mapping)"""
    return move.from_square * 64 + move.to_square



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
    def train_from_file(self, data_file="chess_training_data.pt", epochs=10, batch_size=64):
        print(f"Loading training data from {data_file}...")
        
        # Load data
        data = torch.load(data_file)
        X = data['X']
        y_value = data['y_value']
        y_policy = data['y_policy']
        
        print(f"Loaded {len(X)} positions")
        
        # Simple train/val split
        split = int(0.8 * len(X))
        indices = list(range(len(X)))
        random.shuffle(indices)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                X[indices[:split]], 
                y_value[indices[:split]].unsqueeze(1), 
                y_policy[indices[:split]]
            ), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                X[indices[split:]], 
                y_value[indices[split:]].unsqueeze(1), 
                y_policy[indices[split:]]
            ), 
            batch_size=batch_size
        )
        
        # Training
        device = self.device
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001,weight_decay=1e-4)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_v, batch_p in train_loader:
                batch_X = batch_X.to(device)
                batch_v = batch_v.to(device)
                batch_p = batch_p.to(device)
                
                optimizer.zero_grad()
                policy_out, value_out = self.model(batch_X)
                
                loss = (torch.nn.functional.mse_loss(value_out, batch_v) + 
                    torch.nn.functional.cross_entropy(policy_out, batch_p))
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_v, batch_p in val_loader:
                    batch_X = batch_X.to(device)
                    batch_v = batch_v.to(device)
                    batch_p = batch_p.to(device)
                    
                    policy_out, value_out = self.model(batch_X)
                    val_loss += (torch.nn.functional.mse_loss(value_out, batch_v) + 
                            torch.nn.functional.cross_entropy(policy_out, batch_p)).item()
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Save
        torch.save(self.model.state_dict(), "chess_model.pth")
        print("Model saved to chess_model.pth")
        
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
        print("1. Train model from pre-processed chunks")
        print("2. Load existing model")
        print("3. Play against engine")
        print("4. Watch engine vs random")
        print("5. Watch engine self-play")
        print("6. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            epochs = int(input("Number of epochs (5-20): "))
            batch_size = 64
            
            engine.train_from_file(
                data_file="chess_training_data.pt",
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


if __name__ == "__main__":
    main()