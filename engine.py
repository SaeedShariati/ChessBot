import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
import os
import math
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity   # Skip connection
        out = F.relu(out)
        
        return out

class ChessNet(nn.Module):
    def __init__(self, num_blocks=6):
        super(ChessNet, self).__init__()
        
        # Initial feature extraction
        self.input_conv = nn.Sequential(
            nn.Conv2d(19, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Residual tower
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_blocks)]
        )
        
        # Policy Head
        self.policy = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096)
        )
        
        # Value Head
        self.value = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_layers(x)
        
        return self.policy(x), self.value(x)

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
    
    # Repetition count, ignore
    
    return torch.FloatTensor(tensor)

def move_to_index(move):
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
    
    def ucb_score(self, exploration_constant=1.2):
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
                node = max(node.children.values(), key=lambda c: c.ucb_score())
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
                    value = 1.1
                elif result == "0-1":
                    value = -1.1
                else:
                    value = 0.0
            else:
                # Use neural network for evaluation
                value = self.evaluate(node.board)
            if node.board.turn == chess.WHITE:
                value = -value
            #Backpropagation
            for node in reversed(path):
                node.visits += 1
                node.value_sum += value
                value = -value  # Flip for next parent
        
        return root

    def expand(self, node):
        board = node.board
        
        # Get policy from neural network
        policy, _ = self.predict(board)
        policy_logits = policy[0]
        #mask is used to assign large negative values to policy of illegal moves
        #so that their softmax becomes zero
        mask = torch.full_like(policy_logits, -1e9)
        legal_moves = list(board.legal_moves)
        move_indices = [move_to_index(m) for m in legal_moves]
        mask[move_indices] = 0

        policy_logits = policy_logits + mask
        policy = F.softmax(policy_logits, dim=0)

        for move,idx in zip(legal_moves,move_indices):
            new_board = board.copy()
            new_board.push(move)
            # Get prior probability for this move
            prior = policy[idx].item()
            node.children[move] =  MCTSNode(new_board, parent=node, move=move, prior=prior)
        
    
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
        
        self.mcts = None
    def play_move(self, board, num_simulations=800):
        if self.mcts is None:
            self.mcts = MCTS(self.model, num_simulations)
        root = self.mcts.search(board)
        return self.mcts.get_best_move(root)
    
    def play_game(self, opponent=None, num_moves=200,white = True):
        board = chess.Board()
        self.model.eval()

        for move_num in range(num_moves):
            print(f"\nMove {((move_num+1)/2)}")
            
            
            if board.is_game_over():
                print(f"Game over: {board.result()}")
                b = chess.Board()
                for move in board.move_stack:
                    if(b.turn):
                        print(b.fullmove_number,end='.')
                        print(b.san_and_push(move),end=' ')
                        continue
                    print(b.san_and_push(move))
                    
                break
            
            if board.turn == white:
                move = self.play_move(board)
                print(f"Engine plays: {board.san(move)}")
            else:

                print("Enter your move (e.g., e4):")
                try:
                    move = board.parse_san(input())
                except:
                    print("Invalid")
                    continue
            board.push(move)
            print(board)
        return board
def main():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU found")
    engine = ChessEngine()
    model_path = "chess_model.pth"
    if os.path.exists(model_path):
        engine.model.load_state_dict(torch.load(model_path, map_location=engine.device))
        engine.model.eval()
        print("Model loaded")
    else:
        print("Model not found")
    engine.play_game(opponent="human")


if __name__ == "__main__":
    main()



