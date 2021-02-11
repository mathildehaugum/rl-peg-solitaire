import networkx as nx
import matplotlib.pyplot as plt
from celluloid import Camera
import pyglet


class Visualizer:
    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.camera = Camera(plt.figure())
        self.G, self.pos = self.init_board_visualizer()

    def init_board_visualizer(self):
        G = nx.Graph()
        for current_cell in self.board.get_cells():
            x_pos = current_cell.get_location()[0]  # Positive x-pos to not horizontally flip graph
            y_pos = -current_cell.get_location()[1]  # Negative y-pos to vertically flip graph
            G.add_node(current_cell, pos=(x_pos, y_pos))
            for neighbor_cell in current_cell.get_neighbors():
                G.add_edge(current_cell, neighbor_cell)  # edges can be added before nodes (networkx docs)
        return G, nx.get_node_attributes(G, 'pos')

    def draw_pegs_and_holes(self):
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=self.board.get_holes(),
                               node_color='white', linewidths=2, edgecolors='black')
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=self.board.get_pegs(),
                               node_color='black', linewidths=2, edgecolors="black")
        nx.draw_networkx_edges(self.G, self.pos)

    def draw_state_transition(self, action):
        self.draw_pegs_and_holes()
        if len(action) == 3:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[action[0]],
                                   node_color='green', linewidths=2, edgecolors='black')
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[action[1]],
                                   node_color='red', linewidths=2, edgecolors='black')
            nx.draw_networkx_edges(self.G, self.pos)
        self.camera.snap()  # Creates snapshot of transition

    def draw_board(self):
        self.draw_pegs_and_holes()
        self.camera.snap()

    def animate_visualiser(self):
        animation = self.camera.animate(interval=1000, repeat=False)
        animation.save('images/animation.gif')
        run_pyglet()

    def visualize_episode(self, episode):
        """ Visualizes the performed actions and resulting states in the given episode"""
        for state_action_pair in episode:
            self.draw_state_transition(state_action_pair[1])
            self.player.perform_action(state_action_pair[1])
            self.draw_board()
        self.animate_visualiser()


def run_pyglet():
    # Code for displaying GIF in pyglet application window
    animation = pyglet.image.load_animation('images/animation.gif')
    sprite = pyglet.sprite.Sprite(animation)

    # create a window and set it to the image size
    win = pyglet.window.Window(width=sprite.width, height=sprite.height)

    @win.event
    def on_draw():
        win.clear()
        sprite.draw()

    pyglet.app.run()
