import math
import tensorflow as tf
import numpy as np


class SplitGD:
    """Class for "exposing" the gradients during gradient descent by breaking the call to "fit" into two calls:
    tape.gradient and optimizer.apply_gradients. This enables intermediate modification of the gradients where
    the eligibilities are applied to the gradients before the gradients are use to update the weights"""
    def __init__(self, keras_model, eligibilities, critic_alpha, critic_lambda, critic_gamma, td_error):
        self.model = keras_model
        self.state_eligibilities = eligibilities
        self.learning_rate = critic_alpha
        self.discount_factor = critic_gamma
        self.eligibility_decay = critic_lambda
        self.td_error = td_error  # found and provided by critic

    def update_td_error(self, td_error):
        """ Executed when Neural critic computes new value of td_error"""
        self.td_error = td_error

    def modify_gradients(self, gradients):
        """Step 6a: Modify the gradients to apply the eligibility traces to the weights (w_j = w_j + αδe_j)"""
        for i in range(len(gradients)):
            gradients[i] += self.learning_rate * tf.reshape(self.td_error, [1]) * self.state_eligibilities[i]  # tf.reshape makes td_error shape (1, ) which is necessary to not change shape of gradients, so the shape of gradients is equal the shape of params in fit()
        self.adjust_nn_eligibility(gradients)  # eligibilities are updated after the value function (step 6b)
        return gradients

    def adjust_nn_eligibility(self, gradients):
        """ Step 6b: The eligibilities are updated based on discount factor, eligibility decay and gradients
        (i.e. e(s) <-- γλe(s) + gradients). This represents both eligibility increment and decay for the Neural Critic"""
        for i in range(len(gradients)):
            self.state_eligibilities[i] = (self.discount_factor * self.eligibility_decay * self.state_eligibilities[i]) + gradients[i]

    def gen_loss(self, features, targets, avg=False):
        """ Computes loss for given combination of features (input) and target values (i.e. the forward pass) """
        predictions = self.model(features)  # Feed-forward pass to produce predictions, that are the outputs of the model given the input features
        loss = self.model.loss(targets, predictions)  # predictions are compared to the target to generate the loss
        return tf.reduce_mean(loss).numpy() if avg else loss  # returns the average or computed loss

    def fit(self, features, targets, epochs=1, mbs=1, vfrac=0.1, verbosity=0):
        """Receives tensor and target-value of a state, use these to perform a forward pass to compute the loss function,
        which is given to tape.gradient to produce gradients without applying them. The gradients are modified with the
        eligibilities and then applied to the weights. Goal: separate production and application of gradients"""
        params = self.model.trainable_weights  # get all the weights
        train_inputs, train_targets, val_inputs, val_targets = split_training_data(features, targets, vfrac=vfrac)  # split training data based on validation fraction
        for epoch in range(epochs):
            for _ in range(math.floor(len(train_inputs) / mbs)):  # for each minibatch
                with tf.GradientTape() as tape:  # Use GradientTape to break process in two by getting access to gradients and calling tape.gradient
                    feature_set, target_set = gen_random_minibatch(train_inputs, train_targets, mbs=mbs)  # feature and target set is created by calling helping function with size mbs
                    loss = self.gen_loss(feature_set, target_set, avg=False)  # Forward pass: generate loss of feature-target combination
                    gradients = tape.gradient(loss, params)  # Backward pass: computes gradients of given loss function with respect to given parameters
                    gradients = self.modify_gradients(gradients)  # The reason for doing the split. Gradients need to be modified to include the eligibility traces
                    self.model.optimizer.apply_gradients(zip(gradients, params))  # Apply the modified gradients to the given parameters to update the parameters

            if verbosity > 0:  # Summarize the epoch by printing information about loss, evaluation for training and validation set
                self.end_of_epoch_action(train_inputs, train_targets, val_inputs, val_targets, epoch, verbosity=verbosity)
        # self.gen_evaluation(features, targets, verbosity=2) Turn on to print loss and mean-squared error

    def end_of_epoch_action(self, train_ins, train_targs, valid_ins, valid_targs, epoch, verbosity=1):
        """Method for printing information about loss and mean-squared error. Uses the two methods below"""
        print('\n Epoch: {0}'.format(epoch), end=' ')
        # Calculate Loss and Evaluation for entire training set
        val, loss = self.gen_evaluation(train_ins, train_targs,avg=True,verbosity=verbosity)
        self.status_display(val, loss, verbosity=verbosity, mode='Train')
        val2, loss2 = 0, 0
        if len(valid_ins) > 0:  # Calculate Loss and Evaluation for entire Validation Set
            val2, loss2 = self.gen_evaluation(valid_ins, valid_targs, avg=True, verbosity=verbosity)
            self.status_display(val2,loss2, verbosity=verbosity, mode='Validation')

    def gen_evaluation(self, features, targets, avg=False, verbosity=0):
        loss, evaluation = self.model.evaluate(features, targets, batch_size=len(features), verbose=(1 if verbosity == 2 else 0))
        return evaluation, loss

    def status_display(self, val, loss, verbosity = 1, mode='Train'):
        if verbosity > 0:
            print('{0} *** Loss: {1} Eval: {2}'.format(mode, loss, val), end=' ')


def gen_random_minibatch(inputs, targets, mbs=1):
    """Method for generating random mini batches"""
    indices = np.random.randint(len(inputs), size=mbs)
    return inputs[indices], targets[indices]


def split_training_data(inputs, targets, vfrac=0.1, mix=True):
    """Method for splitting dating into training and validation data used to evaluate the performance of the model """
    vc = round(vfrac * len(inputs))  # vfrac = validation_fraction
    # pairs = np.array(list(zip(inputs,targets)))
    if vfrac > 0:
        pairs = list(zip(inputs, targets))
        if mix: np.random.shuffle(pairs)
        vcases = pairs[0:vc]; tcases = pairs[vc:]
        return np.array([tc[0] for tc in tcases]), np.array([tc[1] for tc in tcases]), \
               np.array([vc[0] for vc in vcases]), np.array([vc[1] for vc in vcases])
        #  return tcases[:,0], tcases[:,1], vcases[:,0], vcases[:,1]  # Can't get this to work properly
    else:
        return inputs, targets, [], []