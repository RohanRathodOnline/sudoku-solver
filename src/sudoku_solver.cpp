#include "sudoku_solver.h"

bool SudokuSolver::solve(std::vector<std::vector<int>>& board) {
    int row = 0;
    int col = 0;

    if (!findEmptyCell(board, row, col)) {
        return true;
    }

    for (int value = 1; value <= 9; ++value) {
        if (!isSafe(board, row, col, value)) {
            continue;
        }

        board[row][col] = value;

        if (solve(board)) {
            return true;
        }

        board[row][col] = 0;
    }

    return false;
}

bool SudokuSolver::isSafe(const std::vector<std::vector<int>>& board, int row, int col, int value) const {
    for (int i = 0; i < 9; ++i) {
        if (board[row][i] == value || board[i][col] == value) {
            return false;
        }
    }

    const int boxStartRow = (row / 3) * 3;
    const int boxStartCol = (col / 3) * 3;

    for (int r = boxStartRow; r < boxStartRow + 3; ++r) {
        for (int c = boxStartCol; c < boxStartCol + 3; ++c) {
            if (board[r][c] == value) {
                return false;
            }
        }
    }

    return true;
}

bool SudokuSolver::findEmptyCell(const std::vector<std::vector<int>>& board, int& row, int& col) const {
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 9; ++c) {
            if (board[r][c] == 0) {
                row = r;
                col = c;
                return true;
            }
        }
    }

    return false;
}
