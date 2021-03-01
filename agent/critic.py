from collections import defaultdict
import random
import tensorflow as tf
import numpy as np
from tensorflow import keras as KER
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import zeros_like
from agent.split_gd import SplitGD


class Critic:
    """Super class for making critic"""
    def __init__(self, critic_alpha, critic_gamma, critic_lambda, is_critic_table):
        self.learning_rate = critic_alpha
        self.discount_factor = critic_gamma
        self.eligibility_decay = critic_lambda
        self.td_error = None
        self.is_critic_table = is_critic_table

    def get_is_critic_table(self):
        """Return true if critic is Table and false if critic is Neural"""
        return self.is_critic_table

    def compute_td_error(self, reward, next_state, current_state):
        """Step 4: TD-error of a state is new value of current_state (r + γV(s') minus the old value (V(s)).
        The get_value method of Table or Neural Critic is used to find the value of the states"""
        self.td_error = (reward + self.discount_factor * self.get_value(next_state)) - self.get_value(current_state)
        if not self.is_critic_table:  # Neural Critic needs to update td_error in split_gd
            self.split_gd.update_td_error(self.td_error)
        return self.td_error


class TableCritic(Critic):
    """Sub class for making table critic"""
    def __init__(self, critic_alpha, critic_gamma, critic_lambda, is_critic_table):
        super().__init__(critic_alpha, critic_gamma, critic_lambda, is_critic_table)
        self.value_function = defaultdict(lambda: random.uniform(0, 1))  # V(s) initialized with small random values
        self.state_eligibilities = defaultdict(lambda: 0)

    def get_value(self, state):
        """The value function gives the predicted value of being in the given state"""
        return self.value_function[state]

    def update_value(self, state):
        """Step 6a: The value function for the state is updated with the product of
        the learning rate, td-error and eligibility of the state (V(s) = V(s)+αδe(s))"""
        self.value_function[state] += self.learning_rate * self.td_error * self.state_eligibilities[state]

    def decay_state_eligibility(self, state):
        """" For each step of the episode the eligibility of all states will decay.
        This method will calculate the decay and is called for each state in the episode """
        self.state_eligibilities[state] *= self.discount_factor * self.eligibility_decay

    def increment_state_eligibility(self, state):
        """" The eligibility of the recent visited state will increment so it can
         contribute in a larger degree to future policy updating """
        self.state_eligibilities[state] = 1

    def reset_eligibilities(self):
        """ Reset eligibility of all states, performed in beginning of each episode"""
        self.state_eligibilities = defaultdict(lambda: 0)


class NeuralCritic(Critic):
    """Sub class for making neural critic"""
    def __init__(self, critic_alpha, critic_gamma, critic_lambda, input_size, hidden_layers_dim, is_critic_table):
        super().__init__(critic_alpha, critic_gamma, critic_lambda, is_critic_table)
        self.value_function_model = self.init_nn(input_size, hidden_layers_dim)  # hidden_layers_dim is list of hidden layers sizes
        self.state_eligibilities = self.reset_eligibilities()  # for NN, eligibilities will affect each weight in network and not states
        self.split_gd = SplitGD(self.value_function_model, self.state_eligibilities, critic_alpha, critic_lambda, critic_gamma, self.td_error)

    def init_nn(self, input_size, hidden_layers_dim):
        """Initializes the neural sequential model by adding layers and compiling the model.
        There is no call to fit(), because the eligibilities need to be applied to the gradients
         before the gradients can be used to update the model weights. This is done in split-gd"""
        opt = Adadelta(learning_rate=self.learning_rate)  # Adagrad is well-suited for dealing with sparse data, Adadelta is extension that solves problem of shrinking learning rate
        loss = MeanSquaredError()  # Larger errors should be penalized more than smaller ones
        model = KER.models.Sequential()
        model.add(KER.layers.Dense(input_size, activation="relu", input_shape=(input_size, )))  # input layer expect one-dimensional array with input_size elements for input. This will automatically build network
        for i in range(len(hidden_layers_dim)):
            model.add(KER.layers.Dense(hidden_layers_dim[i], activation="relu"))  # relu gives quick convergence
        model.add(KER.layers.Dense(1))  # Observation: no activation function gives quicker convergence (could use linear)
        model.compile(optimizer=opt, loss=loss, metrics=["mean_squared_error"])  # MSE is one ot the most preferred metrics for regression tasks
        # model.summary()
        return model

    def get_value(self, state):
        """The value function gives the predicted value of being in the given state. For NN this
        will be the output of the neural model found by giving the state as input. The state is a string
        (e.g. 111011111) and needs to be converted to a tensor before it can be given to the model"""
        tensor_state = convert_state_to_tensor(state)
        return self.value_function_model(tensor_state)

    def update_nn(self, current_state, reward, next_state):
        """The value function of Neural Critic is updated by calling the fit-method of SplitGD.
        This method receives the tensor and target value (i.e. r + γV(s')) of the current state
        and uses these to update the weights of the network to attempt to improve the value function"""
        value_next_state = self.get_value(next_state)
        target_value = reward + self.discount_factor * value_next_state
        tensor_current_state = convert_state_to_tensor(current_state)
        self.split_gd.fit(tensor_current_state, target_value)

    def reset_eligibilities(self):
        """ For Neural Critic, eligibilities are applied to weights that are tensors (i.e. array-like objects).
        To set the eligibilities to value 0, one must retrieve the tensors containing all trainable weights
         and use zeros-like to set all elements in each tensor to 0"""
        state_eligibilities = []
        for params in self.value_function_model.trainable_weights:
            state_eligibilities.append(zeros_like(params))  # https://www.tensorflow.org/api_docs/python/tf/zeros_like
        return state_eligibilities


def convert_state_to_tensor(state):
    """Convert given state from string format to tensor (i.e. array-like object)"""
    state_array = np.array(list(state))  # make numpy array of string
    tensor_state = tf.convert_to_tensor(state_array, np.float32)  # convert numpy array to tensor
    return tf.convert_to_tensor(np.expand_dims(tensor_state, axis=0))  # insert axis on pos 0 to get a tensor shape that corresponds to the input_shape of the neural network (e.g. (15, ) --> (1, 15))
















