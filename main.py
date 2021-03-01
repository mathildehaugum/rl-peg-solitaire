from environment.sim_world import SimWorld
from agent.actor_critic_agent import Agent
from agent.actor import Actor
from agent.critic import TableCritic, NeuralCritic

# TASK 2 TRIANGLE - NN
if __name__ == '__main__':
    #for i in range(1, 20):
        # Initializing sim_world and parameters:
        board_size = 8                                      # 2T: 5             2NN: 5             3T: 4             3NN: 4
        diamond = False                                      # 2T: False         2NN: False         3T: True          3NN: True
        init_holes = [(3, 1), (2, 1), (1, 1)]                                # 2T: [(3,1)]       2NN: [(3,1)]       3T: [(2,1)]/[(1,2)]      3NN: [(2,1)]/[(1,2)]
        sim_world = SimWorld(board_size, diamond, init_holes)
        player = sim_world.get_player()
        board = sim_world.get_board()

        # Initializing actor and parameters:
        actor_alpha = 0.2  # learning rate              # 2T: 0.0005           2NN: 0.0005           3T: 0.7           3NN: 0.7  (step-size in policy update)
        actor_gamma = 0.9  # discount factor                # 2T: 0.9           2NN: 0.9           3T: 0.9           3NN: 0.9  (high gamma --> future rewards are important)
        actor_lambda = 0.9  # eligibility decay (policy)   # 2T: 0.9          2NN: 0.90          3T: 0.85          3NN: 0.9 (reduction in "importance" of SAP in policy update)
        epsilon = 1                                         # 2T: 1             2NN: 1             3T: 1             3NN: 1   (amount of exploring)
        epsilon_decay = 0.1                                # 2T: 0.998         2NN: 0.998         3T: 0.98          3NN: 0.98   (reduction in exploring for each episode)
        actor = Actor(actor_alpha, actor_gamma, actor_lambda, epsilon, epsilon_decay)

        # Initializing critic and parameters:
        is_critic_table = True                             # 2T: True         2NN: False         3T: True          3NN: False
        critic_alpha = 0.001  # learning rate             # 2T: 0.00001       2NN: 0.00001        3T: 0.00001        3NN: 0.00001    (step-size in training = amount update of weights, obs: to small learning rate makes the algorithm go through many iterations to converge, too large can fail to find good solution)
        critic_gamma = 0.9  # discount factor               # 2T: 0.9          2NN: 0.9           3T: 0.9           3NN: 0.9     (high gamma --> future rewards are important)
        critic_lambda = 0.9   # el. decay (value func)     # 2T: 0.9         2NN: 0.90          3T: 0.85          3NN: 0.9    (reduction in "importance" of state in weight update)
        if diamond:
            input_size = board_size * board_size  # equal number of cells in board
        else:
            input_size = 0
            for num in range(1, board_size+1):
                input_size += num
        hidden_layers_dim = [20, 30, 5]
        if is_critic_table:
            critic = TableCritic(critic_alpha, critic_gamma, critic_lambda, is_critic_table)
        else:
            critic = NeuralCritic(critic_alpha, critic_gamma, critic_lambda, input_size, hidden_layers_dim, is_critic_table)

        # Initializing agent and parameters:
        episode_num = 1000                                        # 2T: 1000         2NN: 1000          3T: 200           3NN: 200
        frame_delay = 1000
        agent = Agent(actor, critic, episode_num, sim_world, frame_delay)
        agent.learn()

