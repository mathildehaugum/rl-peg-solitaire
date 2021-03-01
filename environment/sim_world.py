from environment.peg_player import PegPlayer
from environment.peg_board import *


class SimWorld:
    """ Class for creating the PegSolitaire environment and provide RL-agent with necessary information"""
    def __init__(self, size, is_diamond, start_holes):
        if is_diamond:
            self.board = DiamondGrid(size, start_holes)
        else:
            self.board = TriangleGrid(size, start_holes)
        self.player = PegPlayer(self.board)

    def get_board(self):
        """ Returns board """
        return self.board

    def get_player(self):
        """ Returns player """
        return self.player

    def is_neutral_state(self):
        """ Returns true if there are more than one peg on board and at least one available legal action"""
        return self.board.get_cell_nums()[0] > 1 and len(self.get_legal_actions()) > 0

    def is_winning_state(self):
        """ Returns true if there is just one peg on board"""
        return self.board.get_cell_nums()[0] == 1

    def is_losing_state(self):
        """ Returns true if there are more than one peg on board and no available legal actions"""
        return self.board.get_cell_nums()[0] > 1 and len(self.get_legal_actions()) == 0

    def make_state_transition(self, action):
        """ Makes transition between board states by performing given action,
         and returning state and reward produced by performing this action """
        self.player.perform_action(action)
        new_state = self.board.get_binary_state()  # get space efficient binary state to share with RL agent
        new_reward = self.get_reward()  # get reward of action to share with RL agent
        return new_state, new_reward

    def get_legal_actions(self):
        """ Returns the actions that can be performed by the player a list of the tuples,
         where each tuple is a combination of a moving cell, a jumping cell and an empty cell"""
        legal_actions = []
        for current_hole in self.board.get_holes():  # Looks at all hole cells in current board
            hole_row, hole_col = current_hole.get_location()
            hole_neighbors = current_hole.get_neighbors()
            for hole_neighbor in hole_neighbors:  # Checks if hole cell has neighbors that can be jumped over
                neigh_row, neigh_col = hole_neighbor.get_location()
                if not hole_neighbor.get_is_hole():
                    jumping_neighbors = hole_neighbor.get_neighbors()
                    for jumping_neighbor in jumping_neighbors:  # Checks if jumping cell has neighbors that can jump
                        jump_row, jump_col = jumping_neighbor.get_location()
                        if not jumping_neighbor.get_is_hole():
                            if (jump_row == neigh_row == hole_row) or (jump_col == neigh_col == hole_col) or \
                                    ((abs(hole_row-jump_row) == 2) and (abs(hole_col-jump_col) == 2)):
                                legal_actions.append((jumping_neighbor, hole_neighbor, current_hole))
        return legal_actions

    def get_reward(self):
        """ Returns the reward of being in the current board state"""
        reward = 0
        if self.is_losing_state():
            reward -= self.board.get_cell_nums()[0]
        elif self.is_winning_state():
            reward += 1000
        return reward


# if __name__ == '__main__':
#    init_holes = [(1, 0), (0, 1)]
#    diamond = True
#    board_size = 3
#    sim_world = SimWorld(board_size, diamond, init_holes)

#    move_cell = sim_world.get_board().get_cell(0, 2)
#    jump_cell = sim_world.get_board().get_cell(1, 1)
#    hole_cell = sim_world.get_board().get_cell(1, 0)
#    action1 = (move_cell, jump_cell, hole_cell)
#    move_cell = sim_world.get_board().get_cell(2, 2)
#    jump_cell = sim_world.get_board().get_cell(1, 2)
#    hole_cell = sim_world.get_board().get_cell(0, 2)
#    action2 = (move_cell, jump_cell, hole_cell)
#    new_episode = [("state1", action1), ("state2", action2)]
#    visualizer = Visualizer(sim_world.get_board(), sim_world.get_player())
#    visualizer.visualize_episode(new_episode)
#    sim_world.get_player().stringify_action(action2)















