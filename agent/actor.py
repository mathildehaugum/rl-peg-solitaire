from collections import defaultdict
import random


class Actor:
    def __init__(self, actor_alpha, actor_gamma, actor_lambda, epsilon, epsilon_decay):
        self.learning_rate = actor_alpha
        self.discount_factor = actor_gamma
        self.eligibility_decay = actor_lambda
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.td_error = None  # TD-error found by critic
        self.policy = defaultdict(lambda: 0)  # Policy found by actor. Use of defaultdict means that access to non-existing key will add key with default value 0
        self.sap_eligibilities = defaultdict(lambda: 0)  # SAP-based eligibilities found by actor, initialized to 0

    def get_action(self, state, actions):
        """ Given the current state and available actions, return random action or action with highest desirability"""
        legal_actions = actions  # Actions available from the current state
        available_actions = {}  # Key is an available action, while value is the desire to choose this action

        for available_action in legal_actions:
            sap = (state, available_action)
            available_actions.update({available_action: self.policy[sap]})  # if sap ∉ policy, value = 0 because of defaultdict

        if len(available_actions) == 0:
            return None
        else:
            # Normalize values across all legal actions to get probability distribution over possible actions from s:
            # prob_distribution = {key: float(value)/sum(available_actions.values()) for (key, value) in available_actions.items()}
            if self.epsilon >= random.uniform(0, 1):  # Random pick of action (explore)
                return random.choice(list(available_actions))
            else:  # Sort by value to pick action with highest desirability (exploit)
                sorted_available_actions = {key: value for (key, value) in
                                            sorted(available_actions.items(), key=lambda item: item[1], reverse=True)}
                return list(sorted_available_actions)[0]  # Make list of keys and returns the first one

    def update_policy(self, sap):
        """"The policy for the given state-action pair is updated, meaning that the desirability of
        choosing the given action in the given state is updated """
        self.policy[sap] += self.learning_rate * self.td_error * self.sap_eligibilities[sap]

    def decay_sap_eligibility(self, sap):
        """" For each step of the episode the eligibility of all state-action pairs will decay.
        This method will calculate the decay and is called for each state-action pair in the episode """
        self.sap_eligibilities[sap] *= self.discount_factor * self.eligibility_decay

    def increment_sap_eligibility(self, sap):
        """" The eligibility of the recent visited state-action pairs will increment so it can
         contribute in a larger degree to future policy updating """
        self.sap_eligibilities[sap] = 1

    def reset_eligibilities(self):
        """ Reset eligibility of all states """
        self.sap_eligibilities = defaultdict(lambda: 0)

    def update_td_error(self, td_error):
        """ Update td-error with value received from the critic """
        self.td_error = td_error

    def decay_epsilon(self):
        """ Update epsilon to decide degree of exploring/exploiting """
        self.epsilon *= self.epsilon_decay

    def set_epsilon(self, new_value):
        """ Set epsilon to given value, used for setting epsilon = 0 for last episode """
        self.epsilon = new_value



