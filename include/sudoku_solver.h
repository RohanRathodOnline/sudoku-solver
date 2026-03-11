#pragma once

#include <vector>

class SudokuSolver {
public:
    bool solve(std::vector<std::vector<int>>& board);

private:
    bool isSafe(const std::vector<std::vector<int>>& board, int row, int col, int value) const;
    bool findEmptyCell(const std::vector<std::vector<int>>& board, int& row, int& col) const;
};
