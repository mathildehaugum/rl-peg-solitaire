class Cell:
    """ Class for making a peg solitaire piece which is a Cell object"""
    def __init__(self, row, column):
        self.location = (row, column)
        self.name = "Cell" + str(row) + str(column)  # change to piece color
        self.neighbor_list = []
        self.is_hole = False

    def __str__(self):
        """ Changes Cell object representation to string format to make debugging easier"""
        return self.name

    def __repr__(self):
        """ Changes Cell object representation to string format to make debugging easier"""
        return self.name

    def add_neighbor(self, neighbor_cell):
        """ Adds neighbor relationship to both connected cells"""
        self.neighbor_list.append(neighbor_cell)
        neighbor_cell.neighbor_list.append(self)

    def get_neighbors(self):
        """ Gets list of neighbor cells to current cell"""
        return self.neighbor_list

    def get_is_hole(self):
        """ Returns if cell is peg or hole"""
        return self.is_hole

    def set_is_hole(self, value):
        """ Change status of cell to peg (value = false) or hole (value = true)"""
        if isinstance(value, bool):
            self.is_hole = value

    def get_location(self):
        return self.location


