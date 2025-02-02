class PegPlayer:
    def __init__(self, board):
        self.board = board

    def perform_action(self, action):
        """ Performs given action, where the action is a tuple consisting of the moving cell, the jumping cell and the
        empty cell. Action is performed by changing is_hole property of these cells """
        moving_cell, jumping_cell, hole_cell = action
        if None not in {moving_cell, jumping_cell, hole_cell}:
            moving_cell.set_is_hole(True)  # Moving cell becomes a hole, since it jumps "into" the hole
            jumping_cell.set_is_hole(True)  # Jumping cell becomes a hole because its removed
            hole_cell.set_is_hole(False)  # Hole cell becomes a peg, since moving cell jumps "into" it

