import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import clear_output
import time

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

class GameOfLife:
    def __init__(self, board=None, size=(20, 20)):
        """
        Initialize the Game of Life with a given board state or create an empty board of specified size.
        
        Parameters:
        - board: A 2D numpy array or list of lists where 1 represents a live cell and 0 represents a dead cell.
        - size: A tuple (rows, cols) specifying the size of the board if board is not provided.
        """
        if board is not None:
            self.board = np.array(board, dtype=int)
        else:
            self.board = np.zeros(size, dtype=int)
    def __str__(self):
        # Convert the board to a string for printing like 0,1,1,0,1 with a new line for each row
        return "\n".join([",".join(map(str, row)) for row in self.board])
    
    def get_board(self):
        """Return the current board state."""
        return self.board.copy()
    
    def set_board(self, board):
        """Set the board to a new state."""
        self.board = np.array(board, dtype=int)
    
    def random_board(self, density=0.3):
        """Create a random board with the specified density of live cells."""
        rows, cols = self.board.shape
        random_board = np.random.random((rows, cols)) < density
        self.board = random_board.astype(int)
        return self.board.copy()
    
    def count_neighbors(self, row, col):
        """Count the number of live neighbors for a cell at position (row, col)."""
        rows, cols = self.board.shape
        count = 0
        
        # Check all 8 neighboring cells
        for i in range(max(0, row-1), min(rows, row+2)):
            for j in range(max(0, col-1), min(cols, col+2)):
                # Skip the cell itself
                if i == row and j == col:
                    continue
                count += self.board[i, j]
        
        return count
    def next_with_explanation(self):
        """
        Generate the next board state according to Conway's Game of Life rules
        and return a detailed explanation of why each cell changed or remained the same.
        
        Returns:
        - explanation: A string explaining the state change for each cell
        """

        rows, cols = self.board.shape
        next_board = np.zeros_like(self.board)
        explanation_lines = []
        
        # Using triple quotes with no indentation for the intro text
        intro_text = """Alright, let's break down this task step by step.

I'm given a {}x{} board representing the current state of Conway's Game of Life, where 1 represents a live cell and 0 represents a dead cell.
I need to determine the next state of the board based on the rules provided.
For each cell, I'll:
1. Count its live neighbors
2. Apply the rules to determine if it will be alive or dead in the next state
3. State my conclusion

Now, I'll go through each cell one by one, labeling them by their coordinates (row, column), starting from (0, 0) at the top-left.
For each cell, the neighbors are the 8 adjacent cells in all directions (horizontal, vertical, and diagonal). I'll need to be careful at the edges of the board, as cells at the edges have fewer neighbors.
Let's start:
""".format(rows, cols)
    
        explanation_lines.append(intro_text)
        
        for row in range(rows):
            for col in range(cols):
                # Get the current state and count of neighbors
                neighbors = self.count_neighbors(row, col)
                current_cell = self.board[row, col]
                
                # Get the list of neighbor coordinates and their values
                neighbor_coords = []
                neighbor_values = []
                
                for i in range(max(0, row-1), min(rows, row+2)):
                    for j in range(max(0, col-1), min(cols, col+2)):
                        # Skip the cell itself
                        if i == row and j == col:
                            continue
                        neighbor_coords.append((i, j))
                        neighbor_values.append(self.board[i, j])
                
                # Build the neighbor description string
                neighbor_str = ", ".join([f"({i},{j})" for i, j in neighbor_coords])
                
                # Build the neighbor values calculation string
                neighbor_calc = " + ".join([str(val) for val in neighbor_values])
                if neighbor_calc:  # Ensure there's at least one neighbor
                    neighbor_calc += f" = {neighbors}"
                else:
                    neighbor_calc = "0"
                
                # Cell status line
                cell_status = f"Cell ({row},{col}) - Currently {'alive (1)' if current_cell == 1 else 'dead (0)'}"
                
                # Neighbors line
                neighbors_line = f"Neighbors: {neighbor_str}"
                
                # Live neighbors calculation line
                live_neighbors_line = f"Live neighbors: {neighbor_calc}"
                
                # Determine the new state and explanation
                if current_cell == 1:  # Live cell
                    if neighbors < 2:
                        next_board[row, col] = 0  # Dies
                        rule_applied = "This live cell has fewer than 2 live neighbors, so it dies by underpopulation."
                        conclusion = f"Conclusion: Cell ({row},{col}) becomes dead (0)."
                    elif neighbors <= 3:
                        next_board[row, col] = 1  # Stays alive
                        rule_applied = f"This live cell has {neighbors} live neighbors (2 or 3), so it stays alive."
                        conclusion = f"Conclusion: Cell ({row},{col}) remains alive (1)."
                    else:
                        next_board[row, col] = 0  # Dies
                        rule_applied = f"This live cell has more than 3 live neighbors ({neighbors}), so it dies by overpopulation."
                        conclusion = f"Conclusion: Cell ({row},{col}) becomes dead (0)."
                else:  # Dead cell
                    if neighbors == 3:
                        next_board[row, col] = 1  # Becomes alive
                        rule_applied = "This dead cell has exactly 3 live neighbors, so it becomes alive by reproduction."
                        conclusion = f"Conclusion: Cell ({row},{col}) becomes alive (1)."
                    else:
                        next_board[row, col] = 0  # Stays dead
                        rule_applied = f"This dead cell has {neighbors} live neighbors (not exactly 3), so it remains dead."
                        conclusion = f"Conclusion: Cell ({row},{col}) remains dead (0)."
                
                # Combine all lines for this cell
                cell_explanation = f"{cell_status}\n{neighbors_line}\n{live_neighbors_line}\n{rule_applied}\n{conclusion}"
                explanation_lines.append(cell_explanation)
            
        # Update the board
        self.board = next_board
        
        # Compile the explanation
        explanation = "\n\n".join(explanation_lines)
        
        return XML_COT_FORMAT.format(
            reasoning=explanation,
            answer=str(self)
        )

    def next(self):
        """Generate and return the next board state according to Conway's Game of Life rules."""
        rows, cols = self.board.shape
        next_board = np.zeros_like(self.board)
        
        for row in range(rows):
            for col in range(cols):
                neighbors = self.count_neighbors(row, col)
                current_cell = self.board[row, col]
                
                # Apply the rules:
                # 1. Any live cell with fewer than two live neighbors dies (underpopulation)
                # 2. Any live cell with two or three live neighbors lives
                # 3. Any live cell with more than three live neighbors dies (overpopulation)
                # 4. Any dead cell with exactly three live neighbors becomes a live cell (reproduction)
                
                if current_cell == 1:  # Live cell
                    if neighbors == 2 or neighbors == 3:
                        next_board[row, col] = 1  # Cell stays alive
                else:  # Dead cell
                    if neighbors == 3:
                        next_board[row, col] = 1  # Cell becomes alive
        
        self.board = next_board
        return self.board.copy()
    
    def compare(self, other_board):
        """
        Compare the current board state with another board state.
        Returns the number of cells that are different.
        """
        other_board = np.array(other_board, dtype=int)
        
        if self.board.shape != other_board.shape:
            raise ValueError("Boards must have the same shape for comparison")
        
        # Count the number of differences
        return np.sum(self.board != other_board)
    
    def display(self, figsize=(8, 8)):
        """Display the current board state using matplotlib."""
        plt.figure(figsize=figsize)
        cmap = ListedColormap(['white', 'black'])
        plt.imshow(self.board, cmap=cmap)
        plt.grid(True, which='both', color='gray', linewidth=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.title("Conway's Game of Life")
        plt.tight_layout()
        plt.show()

    def compare_with_highlight(self, other_board):
        """
        Compare the current board state with another board state.
        Returns a new board with 1s marking the locations of errors.
        
        Parameters:
        - other_board: A 2D numpy array or list of lists representing another board state
        
        Returns:
        - errors_board: A 2D numpy array where 1 indicates a cell that is different (an error)
                        and 0 indicates matching cells
        """
        other_board = np.array(other_board, dtype=int)
        
        if self.board.shape != other_board.shape:
            raise ValueError("Boards must have the same shape for comparison")
        
        # Create a board highlighting the errors (1 where cells differ, 0 where they match)
        errors_board = (self.board != other_board).astype(int)
        
        return errors_board

    def display_errors(self, other_board, figsize=(10, 10)):
        """
        Compare the current board with another board and display the errors visually.
        
        Parameters:
        - other_board: A 2D numpy array or list of lists representing another board state
        - figsize: Size of the figure to display
        """
        other_board = np.array(other_board, dtype=int)
        errors_board = self.compare_with_highlight(other_board)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Setup color maps
        main_cmap = ListedColormap(['white', 'black'])
        error_cmap = ListedColormap(['white', 'red'])
        
        # Plot current board
        axes[0].imshow(self.board, cmap=main_cmap)
        axes[0].grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)
        axes[0].set_title("Current Board (Correct)")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot other board
        axes[1].imshow(other_board, cmap=main_cmap)
        axes[1].grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)
        axes[1].set_title("Other Board (To Check)")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Plot errors
        axes[2].imshow(errors_board, cmap=error_cmap)
        axes[2].grid(True, which='both', color='gray', linewidth=0.5, alpha=0.5)
        axes[2].set_title(f"Errors (Total: {np.sum(errors_board)})")
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
        return errors_board