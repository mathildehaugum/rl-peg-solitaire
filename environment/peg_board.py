from cell import Cell


class HexagonalGrid:
    """ Class for making a peg solitaire board which is a Hexagonal grid """
    def __init__(self, size):
        self.boardSize = size
        self.board = [[None for i in range(self.boardSize)] for j in range(self.boardSize)]

    def get_cell(self, row, col):
        """ Returns Cell object at given location if it exists """
        if 0 <= row < self.boardSize and 0 <= col < self.boardSize:
            return self.board[row][col]

    def get_cells(self):
        """ Returns all Cell objects from board that is not None (i.e. pegs AND holes) """
        cell_list = []
        for cell_row in self.board:
            for current_cell in cell_row:
                if current_cell is not None:
                    cell_list.append(current_cell)
        return cell_list

    def get_pegs(self):
        """ Returns Cell objects from board that are pegs """
        peg_list = []
        for cell_row in self.board:
            for current_cell in cell_row:
                if current_cell is not None and not current_cell.get_is_hole():
                    peg_list.append(current_cell)
        return peg_list

    def get_holes(self):
        """ Returns Cell objects from board that are holes """
        hole_list = []
        for cell_row in self.board:
            for current_cell in cell_row:
                if current_cell is not None and current_cell.get_is_hole():
                    hole_list.append(current_cell)
        return hole_list

    def get_cell_nums(self):
        """ Returns number of pegs and holes on board """
        cell_list = self.get_cells()  # only contains Cells that are not None
        peg_num = 0
        for current_cell in cell_list:
            if not current_cell.get_is_hole():
                peg_num += 1
        empty_num = len(cell_list) - peg_num
        return peg_num, empty_num

    def reset_board(self):
        """ Reset board state by removing all holes """
        cell_list = self.get_cells()
        for current_cell in cell_list:
            current_cell.set_is_hole(False)

    def init_holes(self, holes):
        """ Create initial board state by placing initial given holes """
        for row, col in holes:
            if self.get_cell(row, col) is not None:
                self.get_cell(row, col).set_is_hole(True)

    def get_binary_state(self):
        """ Returns space efficient and readable binary version of state where peg = 1 and hole = 0 """
        board_state = ""
        for current_cell in self.get_cells():
            if not current_cell.get_is_hole():
                board_state += str(1)
            else:
                board_state += str(0)
        return board_state


class DiamondGrid (HexagonalGrid):
    """ Subclass for creating Diamond shaped Hexagonal grid"""
    def __init__(self, size, holes):
        super().__init__(size)
        self.make_diamond_board()
        self.init_holes(holes)

    def make_diamond_board(self):
        """ Fills diamond board with Cell objects and creates neighborhood following Diamond structure requirements"""
        for r in range(self.boardSize):
            for c in range(self.boardSize):  # avoid redundant calculation by adding neighbors "behind" current cell
                new_cell = Cell(r, c)
                self.board[r][c] = new_cell
                if c > 0:  # add left neighbor-cell
                    new_cell.add_neighbor(self.board[r][c-1])
                if r > 0:  # add above neighbor-cell
                    new_cell.add_neighbor(self.board[r-1][c])
                if r > 0 and c < self.boardSize-1:  # add right diagonal neighbor-cell
                    new_cell.add_neighbor(self.board[r-1][c+1])


class TriangleGrid (HexagonalGrid):
    """ Subclass for creating Triangle shaped Hexagonal grid"""
    def __init__(self, size, holes):
        super().__init__(size)
        self.make_triangle_board()
        self.init_holes(holes)

    def make_triangle_board(self):
        """ Fills triangle board with Cell objects and creates neighborhood following triangle structure requirements"""
        for r in range(self.boardSize):  # avoid redundant calculation by adding neighbors "behind" current cell
            for c in range(r + 1):  # achieve triangle shape
                new_cell = Cell(r, c)
                self.board[r][c] = new_cell
                if c > 0 and self.board[r][c-1] is not None:  # add right neighbor-cell
                    new_cell.add_neighbor(self.board[r][c-1])
                if r > 0 and self.board[r-1][c] is not None:  # add above neighbor-cell
                    new_cell.add_neighbor(self.board[r-1][c])
                if r > 0 and c > 0 and self.board[r-1][c-1] is not None:  # add left diagonal neighbor-cell
                    new_cell.add_neighbor(self.board[r-1][c-1])


"""if __name__ == '__main__':
    init_holes = [(1, 1), (0, 1)]
    diamond = DiamondGrid(4, init_holes)
    print(diamond.get_cells())
    print(diamond.get_holes())
    print(diamond.get_cell_nums())
    diamond.remove_cell(0, 1)
    print(diamond.get_cells())
    print(diamond.get_holes())
    print(diamond.get_cell_nums())
    print(diamond.get_cell(0, 1).get_is_hole())
    diamond.reset_board()
    print(diamond.get_cells())
    print(diamond.get_holes())
    print(diamond.get_cell_nums())"""


