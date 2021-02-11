from collections import defaultdict
import random


class Actor:
    def __init__(self, lr, df, td, epsilon, sim_world):
        self.learning_rate = lr
        self.discount_factor = df
        self.trace_decay = td
        self.epsilon = epsilon
        self.td_error = None  # td-error found by critic
        self.policy = defaultdict(lambda: 0)  # policy found by actor. Values are initially 0 and use of defaultdict means that access to non-existing key will add key with default value 0
        self.sap_eligibilities = defaultdict(lambda: 0)  # SAP-based eligibilities found by actor, initialized to 0
        self.sim_world = sim_world

    def get_action(self, state):
        """ Given the state returns the action that is  """
        legal_actions = self.sim_world.get_legal_actions()  # Actions available from the current state
        available_actions = {}
        for available_action in legal_actions:
            sap = (state, available_action)
            available_actions.update({available_action: self.policy.get(sap)})

        if len(available_actions) == 0:
            return None
        else:
            # Normalize values across all legal actions to get probability distribution over possible actions from s:
            prob_distribution = {key: float(value)/sum(available_actions.values())
                                 for (key, value) in available_actions.items()}
            print(prob_distribution)
            if self.epsilon >= random.uniform(0, 1):
                return random.choice(available_actions)
            else:
                sorted_available_actions = {key: value for (key, value) in
                                            sorted(available_actions.items(), key=lambda item: item[1], reverse=True)}
                return list(sorted_available_actions)[0]  # Make list of keys and returns the first one (i.e. the action with highest value)

    def update_policy(self, sap):
        """"The policy for the given state-action pair is updated, meaning that the probability of
        choosing the given action in the given state is updated """
        self.policy[sap] += self.learning_rate * self.td_error * self.sap_eligibilities[sap]

    def sap_eligibility_decay(self, sap):
        """" For each step of the episode the eligibility of all state-action pairs will decay.
        This method will calculate the decay and is called for each state-action pair in the episode """
        self.sap_eligibilities[sap] *= self.discount_factor * self.trace_decay

    def sap_eligibility_increment(self, sap):
        """" The eligibility of the recent visited state-action pairs will increment so it can
         contribute in a larger degree to future policy updating """
        self.sap_eligibilities[sap] = 1

    def get_sap_eligibilities(self, sap):
        """ Return eligibility of given state-action pair """
        return self.sap_eligibilities[sap]

    def reset_eligibilities(self):
        self.sap_eligibilities = defaultdict(lambda: 0)

    def update_td_error(self, td_error):
        """ Update td-error with value received from the critic """
        self.td_error = td_error

    def update_epsilon(self, new_epsilon):
        """ Update epsilon to decide degree of exploring/exploiting """
        self.epsilon = new_epsilon




