import os
import json
import logging
logger = logging.getLogger(__name__)
import re
import requests
import shutil
from typing import Any
import urllib.parse
from board_to_fen.predict import get_fen_from_image_path
from google import genai
from google.genai import types
from litellm import completion
from smolagents import Tool
from settings import Settings


class BaseCustomTool(Tool):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        
class GetTaskFileTool(BaseCustomTool):
    name = "get_task_file_tool"
    description = """If a file_name is provided, download file associated with a given task_id. Get absolute file path"""
    inputs = {
        "task_id": {"type": "string", "description": "Task ID (required)"},
        "file_name": {"type": "string", "description": "File name (required)"},
    }
    output_type = "string"

    def __init__(self, settings):
        super().__init__(settings)
        self.directory_name = "downloads"
        self.create_dir()
        
    def forward(self, task_id: str, file_name: str) -> str:
        try:
            response = requests.get(f"{self.settings.evaluation_api_base_url}/files/{task_id}", timeout=15)
            response.raise_for_status()
            with open(f"{self.directory_name}/{file_name}", 'wb') as file:
                file.write(response.content)
            return os.path.abspath(f"{self.directory_name}/{file_name}")
        except Exception as e:
            # Fetch the local file instead, dealing with rate limits, etc.
            shutil.copy2(f"files/{file_name}", f"{self.directory_name}/{file_name}")
            return f"{self.directory_name}/{file_name}"
        
    def create_dir(self):
        # Create the directory if it doesn't exist
        if not os.path.exists(self.directory_name):
            os.makedirs(self.directory_name)
            logger.info(f"Directory '{self.directory_name}' created successfully.")
        else:
            logger.debug(f"Directory '{self.directory_name}' already exists.")

class VideoUnderstandingTool(BaseCustomTool):
    name = "VideoUnderstanding"
    description = "Prompt a YouTube video with questions to understand its content."
    inputs = {
        "youtube_url": {"type": "string", "description": "The URL of the YouTube video"},
        "prompt": {"type": "string", "description": "A question or request regarding the video"},
    }
    output_type = "string"

    def __init__(self, settings, model):
        super().__init__(settings)
        self.model = model
        
    def forward(self, youtube_url: str, prompt: str) -> str:
        client = genai.Client(api_key=self.settings.gemini_api_key.get_secret_value())
        try:
            video_description = client.models.generate_content(
                model=self.model,
                contents=types.Content(
                    parts=[
                        types.Part(
                            file_data=types.FileData(file_uri=youtube_url)
                        ),
                        types.Part(text=prompt)
                    ]
                )
            )
            return video_description.text
        except Exception as e:
            logger.error(f"Error understanding video: {e}")
            return False

class AudioUnderstandingTool(BaseCustomTool):
    name = "AudioUnderstanding"
    description = "Prompt a local audio file with questions to understand its content."
    inputs = {
        "file_path": {"type": "string", "description": "The local file of the audio"},
        "prompt": {"type": "string", "description": "A question or request regarding the audio"},
    }
    output_type = "string"

    def __init__(self, settings, model):
        super().__init__(settings)
        self.model = model

    def forward(self, file_path: str, prompt: str) -> str:
        client = genai.Client(api_key=self.settings.gemini_api_key.get_secret_value())
        try:
            mp3_file = client.files.upload(file=f"{file_path}")
            audio_description = client.models.generate_content(
                model=self.model,
                contents=[prompt, mp3_file]
            )
            return audio_description.text
        except Exception as e:
            logger.error(f"Error understanding audio: {e}")
            return False

class ConvertChessMoveTool(BaseCustomTool):
    name = "ConvertChessMove"
    description = "Convert a chess move from coordinate notation to algebraic notation."
    inputs = {
        "piece_placement": {"type": "string", "description": "The chess piece placement in plain text"},
        "move": {"type": "string", "description": "The move in coordinate notation (e.g., e2e4)"},
    }
    output_type = "string"

    def __init__(self, settings, model):
        super().__init__(settings)
        self.model = model

    def forward(self, piece_placement: str, move: str) -> str:
        move_message = (
            f"Convert this chess move from coordinate notation to algebraic "
            f"notation: {move}. Use the following {piece_placement}. Do not provide any additional "
            "thinking or commentary in the response, the algebraic notation only."
            )
        messages = [{ "content": move_message, "role": "user"}]
        response = completion(
                    model=self.model, 
                    temperature=0.0,
                    messages=messages,
                    api_key=self.settings.openrouter_api_key.get_secret_value()
                )
        return response.choices[0].message.content

class BestChessMoveTool(BaseCustomTool):
    name = "BestChessMove"
    description = "Get best chess move in coordinate notation based on a FEN representation."
    inputs = {
        "fen": {"type": "string", "description": "The FEN (Forsyth-Edwards Notation) \
                representation of the chess position. Example \
                rn1q1rk1/pp2b1pp/2p2n2/3p1pB1/3P4/1QP2N2/PP1N1PPP/R4RK1 b - - 1 11"},
    }
    output_type = "string"

    def forward(self, fen: str) -> str:
        try:
            url = f"{self.settings.chess_eval_url}?fen={urllib.parse.quote(fen)}&depth=15"
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and json.loads(response.text)['success'] == True:
                return json.loads(response.text)['bestmove'].split()[1]
            else:
                raise ValueError(f"Error getting chess evaluation: {response.status_code}")
        except Exception as e:
            logger.error(f"Error getting chess evaluation: {e}")

class ChessBoardFENTool(Tool):
    name = "ChessBoardFEN"
    description = "Get the FEN representation from an image of a chess board and a player turn."
    inputs = {
        "image_path": {"type": "string", "description": "The local file of the chess board image"},
        "player_turn": {"type": "string", 
                "description": "The player with the next turn in the match, must be 'w' or 'b'"}
    }
    output_type = "string"
    
    def _expand_fen_rank(self, rank_str):
        """
        Expands a single rank string from FEN notation (e.g., 'p2b4')
        into a list of 8 characters representing the squares.
        Uses ' ' for empty squares.
        """
        expanded_rank = []
        for char in rank_str:
            if char.isdigit():
                # Add number of empty squares specified by the digit
                expanded_rank.extend([' '] * int(char))
            else:
                # Add the piece character
                expanded_rank.append(char)
        # Validate rank length
        if len(expanded_rank) != 8:
            raise ValueError(f"Invalid FEN rank string (length != 8): {rank_str}")
        return expanded_rank

    def _compress_fen_rank(self, rank_list):
        """
        Compresses a list of 8 characters (representing a rank)
        back into FEN rank notation (e.g., turns [' ', 'K', ...] into '1K6').
        Assumes ' ' represents an empty square.
        """
        if len(rank_list) != 8:
            raise ValueError(f"Invalid rank list (length != 8): {rank_list}")

        compressed_rank = ""
        empty_count = 0
        for char in rank_list:
            if char == ' ':
                empty_count += 1
            else:
                # If we encountered a piece after empty squares, add the count
                if empty_count > 0:
                    compressed_rank += str(empty_count)
                    empty_count = 0
                # Add the piece
                compressed_rank += char
        # If the rank ends with empty squares, add the final count
        if empty_count > 0:
            compressed_rank += str(empty_count)
        return compressed_rank

    def _invert_mirror_fen(self, fen_string):
        """
        Takes a FEN string, inverts the board vertically, mirrors it horizontally,
        and returns the new FEN string representing this transformed view.
        The other FEN fields (turn, castling, etc.) are preserved.
        """
        try:
            # 1. Split FEN into parts
            parts = fen_string.strip().split(' ')
            if len(parts) != 6:
                raise ValueError("FEN string must have 6 space-separated fields.")
            board_part = parts[0]
            other_parts = parts[1:] # Side-to-move, castling, ep, halfmove, fullmove

            # 2. Parse the board part into an 8x8 representation
            rank_strings = board_part.split('/')
            if len(rank_strings) != 8:
                raise ValueError("FEN board part must have 8 ranks separated by '/'.")

            # original_board[0] corresponds to rank 8, original_board[7] to rank 1
            original_board = [self._expand_fen_rank(r) for r in rank_strings]

            # 3. Create a new empty 8x8 board for the transformed state
            # Using ' ' as the placeholder for empty squares
            transformed_board = [[' ' for _ in range(8)] for _ in range(8)]

            # 4. Apply the inversion (vertical flip) and mirror (horizontal flip)
            for r in range(8): # Iterate through original rows (ranks 8 down to 1)
                for c in range(8): # Iterate through original columns (files a to h)
                    # The piece at original [r][c] moves to transformed [7-r][7-c]
                    transformed_board[7 - r][7 - c] = original_board[r][c]

            # 5. Generate the new FEN board string from the transformed board
            # Read ranks from top (index 0 = rank 8) to bottom (index 7 = rank 1)
            new_rank_strings = [self._compress_fen_rank(row) for row in transformed_board]
            new_board_part = "/".join(new_rank_strings)

            # 6. Reassemble the full FEN string
            return " ".join([new_board_part] + other_parts)

        except Exception as e:
            # Return error message if parsing or processing fails
            return f"Error processing FEN: {e}. Input: '{fen_string}'"

    def _add_fen_game_state(self, board_placement,
                        side_to_move,
                        castling="-",
                        en_passant="-",
                        halfmove_clock=0,
                        fullmove_number=1):
        """
        Appends standard game state information to a FEN board placement string.

        Args:
            board_placement (str): The board layout part of the FEN string
                                (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR").
            side_to_move (str): The active color ('w' for White, 'b' for Black).
                                Case-insensitive, will be converted to lowercase.
            castling (str, optional): Castling availability string (e.g., "KQkq", "-").
                                    Defaults to "-".
            en_passant (str, optional): En passant target square string (e.g., "e3", "-").
                                        Defaults to "-".
            halfmove_clock (int, optional): The number of halfmoves since the last
                                        capture or pawn advance. Defaults to 0.
            fullmove_number (int, optional): The number of the full move. Starts at 1
                                        and increments after Black's move. Defaults to 1.

        Returns:
            str: The complete FEN string including the game state,
                or an error message string if inputs are invalid.
        """
        # Validate side_to_move
        side_to_move_lower = str(side_to_move).lower()
        if side_to_move_lower not in ['w', 'b']:
            return f"Error: side_to_move must be 'w' or 'b', received '{side_to_move}'"

        # Validate clock values (should be non-negative integers, fullmove >= 1)
        try:
            halfmove_clock = int(halfmove_clock)
            fullmove_number = int(fullmove_number)
            if halfmove_clock < 0:
                raise ValueError("halfmove_clock cannot be negative.")
            if fullmove_number < 1:
                raise ValueError("fullmove_number must be 1 or greater.")
        except (ValueError, TypeError):
            return (f"Error: halfmove_clock ('{halfmove_clock}') and "
                    f"fullmove_number ('{fullmove_number}') must be valid integers "
                    f"(non-negative and positive respectively).")

        # Assemble the full FEN string using the validated/defaulted values
        # Note: castling and en_passant strings are used directly as passed or defaulted.
        # More complex validation could be added for them if needed.
        full_fen = (f"{board_placement} {side_to_move_lower} {castling} "
                    f"{en_passant} {halfmove_clock} {fullmove_number}")

        return full_fen

    def forward(self, image_path: str, player_turn: str) -> str:
        board_placement = get_fen_from_image_path(image_path)
        
        #  Inversion makes board_to_fen output Stockfish compatible
        board_fen = self._add_fen_game_state(board_placement, player_turn)
        board_fen_inverted = self._invert_mirror_fen(board_fen) 
        
        return board_fen_inverted
    