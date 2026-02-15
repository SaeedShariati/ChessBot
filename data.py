# filter_chess_data_streaming.py
import zstandard as zstd
import chess
import chess.pgn
import io
import torch
import numpy as np
from tqdm import tqdm
import random
import os

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

def process_file_streaming(zst_path, min_elo=1500, samples_per_game=10, 
                          max_positions=500000, output_prefix="chess_data"):
    """
    Process file in streaming mode - NEVER stores all positions in memory!
    Saves directly to disk in chunks
    """
    print(f"Processing {zst_path} in STREAMING mode...")
    
    chunk_size = 10000  # Save every 10k positions
    positions_processed = 0
    current_chunk = []
    chunk_num = 0
    
    # Open the compressed file
    with open(zst_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            pbar = tqdm(desc="Processing games")
            games_processed = 0
            games_kept = 0
            
            while positions_processed < max_positions:
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
                        
                        # Sample positions from this game
                        moves = list(game.mainline_moves())
                        if moves:
                            # Take up to samples_per_game random positions
                            sample_indices = random.sample(
                                range(len(moves)), 
                                min(samples_per_game, len(moves))
                            )
                            
                            for i, move_idx in enumerate(sorted(sample_indices)):
                                # Replay to this position
                                temp_board = game.board()
                                for move in moves[:move_idx + 1]:
                                    temp_board.push(move)
                                
                                # Convert to tensor
                                tensor = board_to_tensor(temp_board)
                                current_chunk.append({
                                    'tensor': tensor,
                                    'game_id': f"{games_kept}_{i}",
                                    'white_elo': white_elo,
                                    'black_elo': black_elo,
                                    'result': game.headers.get('Result', '*')
                                })
                                positions_processed += 1
                                
                                # Save chunk when full
                                if len(current_chunk) >= chunk_size:
                                    save_chunk(current_chunk, chunk_num, output_prefix)
                                    chunk_num += 1
                                    current_chunk = []
                                
                                if positions_processed >= max_positions:
                                    break
                    
                    pbar.set_description(f"Kept: {games_kept} games, {positions_processed} positions")
                    
                except Exception as e:
                    continue
            
            pbar.close()
    
    # Save final chunk
    if current_chunk:
        save_chunk(current_chunk, chunk_num, output_prefix)
    
    print(f"\nComplete! Saved {positions_processed} positions in {chunk_num + 1} chunks")
    return chunk_num + 1

def save_chunk(chunk, chunk_num, output_prefix):
    """Save a chunk of positions to disk"""
    filename = f"{output_prefix}_chunk_{chunk_num:04d}.pt"
    
    # Extract tensors and metadata
    tensors = torch.stack([item['tensor'] for item in chunk])
    metadata = [{k: v for k, v in item.items() if k != 'tensor'} for item in chunk]
    
    # Save
    torch.save({
        'tensors': tensors,
        'metadata': metadata
    }, filename)
    
    print(f"\nSaved chunk {chunk_num} with {len(chunk)} positions to {filename}")

def create_dataset_loader(output_prefix, batch_size=64):
    """
    Create a PyTorch DataLoader that streams from saved chunks
    """
    import glob
    
    class ChessDataset(torch.utils.data.IterableDataset):
        def __init__(self, chunk_files):
            self.chunk_files = chunk_files
        
        def __iter__(self):
            for chunk_file in self.chunk_files:
                data = torch.load(chunk_file)
                for i, tensor in enumerate(data['tensors']):
                    yield tensor, data['metadata'][i]
    
    chunk_files = sorted(glob.glob(f"{output_prefix}_chunk_*.pt"))
    dataset = ChessDataset(chunk_files)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    return loader

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    MIN_ELO = 1500
    MAX_POSITIONS = 500000  # Stop after 500k positions (adjust based on your needs)
    
    files_to_process = [
        "lichess_db_standard_rated_2015-01.pgn.zst",
        "lichess_db_standard_rated_2013-01.pgn.zst",
    ]
    
    for file in files_to_process:
        if os.path.exists(file):
            process_file_streaming(
                file,
                min_elo=MIN_ELO,
                max_positions=MAX_POSITIONS // len(files_to_process),
                output_prefix=f"chess_{file[:4]}_{MIN_ELO}elo"
            )
    
    print("\n✅ Streaming processing complete!")
    print(f"Data saved in chunks: chess_*_{MIN_ELO}elo_chunk_*.pt")