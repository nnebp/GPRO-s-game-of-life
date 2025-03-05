import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import clear_output
import time

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