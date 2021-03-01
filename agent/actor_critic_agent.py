# Code performing the steps of the actor-critic algorithm by calling methods in actor and critic
from visualization import Visualizer
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, actor, critic, episode_num, sim_world, visualization_speed):
        self.actor = actor
        self.critic = critic
        self.episode_num = episode_num
        self.sim_world = sim_world
        self.visualizer = Visualizer(self.sim_world.get_board(), self.sim_world.get_player(), visualization_speed)

    def learn(self):
        """Runs the steps of the actor-critic algorithm for each episode """
        # The value function of the critic and the policy of the actor is inilialized in critic and actor respectively
        plot_episode_nums = []
        plot_num_pegs_left = []

        for episode in range(self.episode_num):

            # Last episode should exploit and not explore, so epsilon is set to 0
            if episode == self.episode_num-1:
                self.actor.set_epsilon(0)
            else:
                self.actor.decay_epsilon()

            # List of tuples, where each tuple gives (state, action, reward, next_state)
            current_episode_steps = []

            # Reset eligibilities in actor and critic
            self.actor.reset_eligibilities()
            self.critic.reset_eligibilities()

            # Initialise state and action
            current_state = self.sim_world.get_board().get_binary_state()
            legal_actions = self.sim_world.get_legal_actions()
            current_action = self.actor.get_action(current_state, legal_actions)

            # Repeat for each step of the episode
            while self.sim_world.is_neutral_state():

                # Step 1-2: perform action, find next state and receive reward
                next_state, reward = self.sim_world.make_state_transition(current_action)
                next_legal_actions = self.sim_world.get_legal_actions()
                next_action = self.actor.get_action(next_state, next_legal_actions)

                # Step 3: Actor increment eligibility of visited SAP
                self.actor.increment_sap_eligibility((current_state, current_action))

                # Step 4: Critic compute td-error for current state (δ = r + γV(s') - V(s))
                td_error = self.critic.compute_td_error(reward, current_state, next_state)
                self.actor.update_td_error(td_error)  # Actor receives TD-error from critic

                # Step 5: Table Critic increment eligibility of visited state.
                # Neural Critic use weight-gradients to increment eligibility of often visited states (see split_gd.py)
                if self.critic.get_is_critic_table():
                    self.critic.increment_state_eligibility(current_state)

                # Step 6: Value function of critic, policy of actor and eligibilites are updated for each SAP in episode
                for step in current_episode_steps:
                    sap = step[0:2]  # step = (current_state, current_action, reward, next_state)

                    # Update value function and decay eligibility of critic
                    if self.critic.get_is_critic_table():
                        self.critic.update_value(sap[0])
                        self.critic.decay_state_eligibility(sap[0])
                    else:
                        self.critic.update_nn(sap[0], step[2], step[3])  # reward and next_state used to find target value

                    # Update policy and decay eligibility for actor
                    self.actor.update_policy(sap)
                    self.actor.decay_sap_eligibility(sap)

                # Found action, reward and transition of current state is saved for further progression in episode
                current_episode_steps.append((current_state, current_action, reward, next_state))
                current_state = next_state
                current_action = next_action

            # Print and plot the result of each episode
            if len(current_episode_steps) > 0:
                print("Episode " + str(episode) + " achieves " + str(current_episode_steps[len(current_episode_steps)-1][2]) + " points.")
                plot_episode_nums.append(episode)
                plot_num_pegs_left.append(self.sim_world.get_board().get_cell_nums()[0])

            self.sim_world.get_board().reset_board()

            # Call visualize_episode for last episode
            if episode == self.episode_num-1:
                print("Episode " + str(episode) + " achieves " + str(current_episode_steps[len(current_episode_steps)-1][2]) + " points.")
                self.visualizer.visualize_episode(current_episode_steps)
                print("Game visualization finished")

        print("Plotting")
        plt.plot(plot_episode_nums, plot_num_pegs_left)
        plt.savefig('images/learning_plot.png')
