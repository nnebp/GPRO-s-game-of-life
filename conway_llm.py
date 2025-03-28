import numpy as np
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from game_of_life import GameOfLife

def generate_next_state_with_llm(game, model, tokenizer, device=None, max_new_tokens=600):
    """
    Generate the next state of the Game of Life using an LLM.
    The LLM is prompted to reason through each cell and then provide the final board.
    
    Args:
        game: The GameOfLife object containing the current board
        model: The LLM model
        tokenizer: The tokenizer for the model
        device: The device to run on (optional)
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        next_board: The predicted next board state (2D numpy array)
    """
    import numpy as np
    
    # Get the current board from the GameOfLife object
    board = game.get_board()
    rows, cols = board.shape
    
    # Convert board to string representation
    board_str = "\n".join(" ".join(str(cell) for cell in row) for row in board)
    
    # Create detailed prompt with instructions for cell-by-cell reasoning
    prompt = f"""You are tasked with predicting the next state of Conway's Game of Life.
    
The rules of Conway's Game of Life are:
1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors survives
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction)

Current board state:
{board_str}

For each cell, I want you to:
1. Count how many live neighbors it has
2. Apply the rules to determine if it will be alive (1) or dead (0) in the next state
3. Clearly state your conclusion for that cell

After reasoning through all cells, please output the complete next board state as a grid of 0s and 1s, with the exact same dimensions ({rows}x{cols}).

Begin your analysis by examining each cell, and then end with a clearly formatted final board.

Format your final answer as follows:
FINAL BOARD:
[board representation with only 0s and 1s, no other characters]
END BOARD

Now, proceed with your cell-by-cell analysis:
"""

    # Generate prediction from the model
    inputs = tokenizer(prompt, return_tensors="pt")
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    output = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        temperature=0.1,  # Lower temperature for more deterministic outputs
        do_sample=True
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the final board from the response
    try:
        # Find the board between "FINAL BOARD:" and "END BOARD"
        if "FINAL BOARD:" in response and "END BOARD" in response:
            board_section = response.split("FINAL BOARD:")[1].split("END BOARD")[0].strip()
            
            # Parse the board
            next_board = parse_board_from_text(board_section, rows, cols)
            return next_board
        else:
            print("Couldn't find the expected markers in the response. Attempting to parse the end of the response...")
            # Try to extract the last section that might contain the board
            lines = response.strip().split('\n')
            end_section = '\n'.join(lines[-rows-5:]) if len(lines) > rows else response
            next_board = parse_board_from_text(end_section, rows, cols)
            return next_board
    except (IndexError, ValueError) as e:
        print(f"Error extracting board from response: {e}")
        print("Full response from model:")
        print(response)
        # Return a copy of the original board as a fallback
        return board.copy()

def parse_board_from_text(text, expected_rows, expected_cols):
    """
    Parse a board from text format into a 2D numpy array.
    This is robust to various delimiters and whitespace.
    
    Args:
        text: String representation of the board
        expected_rows: Expected number of rows
        expected_cols: Expected number of columns
        
    Returns:
        board: 2D numpy array representing the board
    """
    import numpy as np
    import re
    
    # Remove any non-relevant characters, keeping only 0s, 1s, and whitespace
    clean_text = re.sub(r'[^01\s]', '', text)
    
    # Split into rows
    rows = [row for row in clean_text.strip().split('\n') if row.strip()]
    
    # Create the board
    board = np.zeros((expected_rows, expected_cols), dtype=int)
    
    # Try to parse the text into the board
    for i, row_text in enumerate(rows):
        if i >= expected_rows:
            break
            
        # Extract just the 0s and 1s
        digits = re.findall(r'[01]', row_text)
        
        for j, digit in enumerate(digits):
            if j >= expected_cols:
                break
            board[i, j] = int(digit)
    
    # Check if the board has the expected dimensions
    if len(rows) != expected_rows or any(len(re.findall(r'[01]', row)) != expected_cols for row in rows[:expected_rows]):
        print(f"Warning: Parsed board dimensions ({len(rows)}x{len(re.findall(r'[01]', rows[0]) if rows else 0)}) "
              f"don't match expected dimensions ({expected_rows}x{expected_cols})")
    
    return board

def board_to_text(board):
    """Convert a Game of Life board to a text representation."""
    rows, cols = board.shape
    text = ""
    for i in range(rows):
        for j in range(cols):
            text += "1" if board[i, j] == 1 else "0"
        text += "\n"
    return text.strip()

def board_to_visual(board):
    """Convert a Game of Life board to a more visual text representation."""
    rows, cols = board.shape
    text = f"{rows}x{cols} board:\n"
    for i in range(rows):
        for j in range(cols):
            text += "■" if board[i, j] == 1 else "□"
        text += "\n"
    return text.strip()

def parse_board_from_output(output_text, shape):
    """Parse a board of 0s and 1s from the model output text."""
    rows, cols = shape
    
    # Extract potential board lines (lines containing 0s and 1s)
    lines = output_text.strip().split('\n')
    board_lines = []
    
    for line in lines:
        # Keep only 0s and 1s
        cleaned_line = ''.join(c for c in line if c in '01')
        if len(cleaned_line) > 0:
            board_lines.append(cleaned_line)
    
    # If we found lines that could be a board
    if len(board_lines) >= rows:
        # If any line is too short, pad it
        for i in range(len(board_lines)):
            if len(board_lines[i]) < cols:
                board_lines[i] = board_lines[i] + '0' * (cols - len(board_lines[i]))
        
        # Create the board from the lines
        board = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            if i < len(board_lines):
                line = board_lines[i]
                for j in range(min(cols, len(line))):
                    if line[j] == '1':
                        board[i, j] = 1
        
        return board
    
    # If we couldn't find enough lines, try to extract all digits
    digits = ''.join(c for c in output_text if c in '01')
    
    if len(digits) >= rows * cols:
        # Create the board from the sequence of digits
        board = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(digits) and digits[idx] == '1':
                    board[i, j] = 1
        
        return board
    
    # If we still don't have enough digits, return None
    return None

#TODO: delete this
def generate_next_state_with_llm_old(game, model, tokenizer, device='cuda'):
    """Use an LLM to predict the next state of the Game of Life board."""
    current_board = game.get_board()
    board_text = board_to_text(current_board)
    board_visual = board_to_visual(current_board)
    
    # Create a prompt that explains the task
    prompt = f"""Conway's Game of Life is a cellular automaton where cells live or die based on their neighbors.

Rules:
1. Any live cell with fewer than two live neighbors dies (underpopulation)
2. Any live cell with two or three live neighbors lives
3. Any live cell with more than three live neighbors dies (overpopulation)
4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction)

Current board state:
{board_visual}

Representation as 0s and 1s (0=dead, 1=alive):
{board_text}

Calculate the next state by applying the Game of Life rules to each cell.
Count each cell's live neighbors (horizontally, vertically, and diagonally adjacent).
Apply the rules to determine if it lives or dies in the next generation.

Output ONLY the next state as a grid of 0s and 1s with the same dimensions:
"""

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=500,
            do_sample=False,
            temperature=0.1,
        )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract the part after the prompt
    generated_part = output_text[len(prompt):].strip()
    
    # Parse the board from the output
    predicted_board = parse_board_from_output(generated_part, current_board.shape)
    
    # If parsing failed, return a board of zeros
    if predicted_board is None:
        print("Warning: Could not parse a valid board from the model output. Returning empty board.")
        predicted_board = np.zeros_like(current_board)
    
    return predicted_board

def load_model_and_tokenizer(model_name, quantize=True):
    """Load a model and tokenizer with appropriate settings."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer with padding on the left
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization if requested
    if quantize:
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            print("Model loaded with 4-bit quantization")
        except (ImportError, Exception) as e:
            print(f"Quantization failed: {e}")
            # Fall back to regular loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Model loaded with float16 precision")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded with float16 precision")
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Conway's Game of Life with LLM next state prediction")
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1',
                        help='HuggingFace model to use. Default: mistralai/Mistral-7B-v0.1')
    parser.add_argument('--size', type=int, default=5,
                        help='Size of the Game of Life board (square). Default: 5x5')
    parser.add_argument('--rows', type=int, default=None, 
                        help='Number of rows for the Game of Life board. Overrides --size')
    parser.add_argument('--cols', type=int, default=None,
                        help='Number of columns for the Game of Life board. Overrides --size')
    parser.add_argument('--density', type=float, default=0.3,
                        help='Density of live cells in the random board. Default: 0.3')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable 4-bit quantization')
    
    args = parser.parse_args()
    
    # Determine board dimensions
    rows = args.rows if args.rows is not None else args.size
    cols = args.cols if args.cols is not None else args.size
    
    # Create a Game of Life instance with a random board
    game = GameOfLife(size=(rows, cols))
    game.random_board(density=args.density)
    
    # Display the initial board
    print("Initial board:")
    game.display()
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, quantize=not args.no_quantize)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else  "mps" if torch.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Store the initial board
    initial_board = game.get_board()
    
    # Calculate the correct next state using the built-in method
    correct_next_state = game.next()
    
    # Reset the board to the initial state
    game.set_board(initial_board)
    
    # Use the LLM to predict the next state
    print("Generating next state with LLM...")
    predicted_next_state = generate_next_state_with_llm(game, model, tokenizer, device)
    
    # Create instances for display
    predicted_game = GameOfLife(board=predicted_next_state)
    correct_game = GameOfLife(board=correct_next_state)
    
    # Display the predicted next state
    #print("Predicted next state:")
    #predicted_game.display()
    
    # Display the correct next state
    #print("Correct next state (using built-in rules):")
    #correct_game.display()
    
    # Compare the results
    print("Comparing results...")
    correct_game.display_errors(predicted_next_state)
    
    # Calculate accuracy
    total_cells = np.prod(initial_board.shape)
    differences = np.sum(correct_next_state != predicted_next_state)
    accuracy = (total_cells - differences) / total_cells * 100
    
    print(f"Accuracy: {accuracy:.2f}% ({total_cells - differences}/{total_cells} cells correct)")



if __name__ == "__main__":
    main()