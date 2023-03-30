import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import numba
import warnings
from numba import jit
import sys

sys.path.append("..")

warnings.simplefilter('ignore', category=numba.errors.NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=numba.errors.NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=numba.errors.NumbaWarning)

@jit
def discount_with_dones(rewards, dones, state_values, gamma):

    #find where the dones are
    done_locations = np.append(np.zeros(1), np.where(np.array(dones)==0)[0]) #0 means terminal state

    discounted_all = []

    #We need to cycle through the locations
    for m in range(len(done_locations)-1):
      discounted = []
      start_ind = int(done_locations[m])
      end_ind = int(done_locations[m+1])

      v_s_ =state_values[end_ind]
      for reward, done in zip(rewards[start_ind:end_ind][::-1], dones[start_ind:end_ind][::-1]):
          v_s_ = reward + gamma*v_s_*(done)
          discounted.append(v_s_)
      discounted.reverse()
      discounted_all+=discounted

    discounted_all.append(rewards[-1]) #add final reward

    return discounted_all

@jit
def generalized_advantage_estimate(
    gamma, lamda, value_old_state, value_new_state, reward, done
):
    """
    Get generalized advantage estimate of a trajectory
    gamma: trajectory discount (scalar)
    lamda: exponential mean discount (scalar)
    value_old_state: value function result with old_state input
    value_new_state: value function result with new_state input
    reward: agent reward of taking actions in the environment
    done: flag for end of episode
    """
    value_old_state = np.array(value_old_state)
    value_new_state = np.array(value_new_state)
    value_new_state = np.append(value_new_state,0)
    reward = np.array(reward)
    done = np.array(done)
    #print('gamma {}, lambda {}, value old state {}, value new state {}, rewards {}, done {}'.format(
    #gamma, lamda, value_old_state.shape, value_new_state.shape, reward.shape, done.shape    
    #))
    batch_size = len(reward)

    advantage = np.zeros(batch_size + 1)

    for t in reversed(range(batch_size)):
        #print(t)
        delta = reward[t] + (gamma * value_new_state[t] * done[t]) - value_old_state[t]
        advantage[t] = delta + (gamma * lamda * advantage[t + 1] * done[t])

    value_target = advantage[:batch_size] + np.squeeze(value_old_state)

    return advantage[:batch_size], value_target

def reshape_gradient(gradient, weights):
    updated_gradients = []
    offset = 0

    for (i, itm) in enumerate(weights):
        new_shape = itm.shape
        chunk = np.prod(new_shape)

        if i == len(weights):
            new_grad = gradient[offset + chunk:]
        else:
            new_grad = gradient[offset:offset + chunk]
        offset = chunk
        new_grad = tf.reshape(new_grad, new_shape)
        updated_gradients.append(tf.constant(new_grad))

    return updated_gradients

def return_targets(reward_history,state_values,gamma):
    targets = []
    for ind in range(len(reward_history)-1):
        targets.append(reward_history[ind]+gamma*state_values[ind+1])
    targets.append(reward_history[-1])
    return targets

def SteinUpdateStep(weights_list, gradient_list,
                    num_agents=2,
                    temp=10.0):
    """
    Inputs:
    (1) Numpy array: Extracted weights from all agents for current batch
    (2) Numpy array: Calculated gradients from all agents for current batch (list of 1D concatenated array)
    (3) temp: Stein update temperature
    Output:
    (1) Numpy array: Stein update gradient. This should then be applied with a choice of optimizer.
    """

    # we have a list of list of weights and a list of gradients
    # we need to first iterate through them all and flatten them down

    flat_weights = []
    flat_grads = []
    #shape_list = [itm.shape for itm in weights_list[0]]

    for weight in weights_list:
        flat_weights.append(
            np.concatenate([vec.numpy().flatten() if type(vec) != np.ndarray else vec.flatten() for vec in weight]))
    #print('Weights list length is '.format(len(weights_list)))

    for grad in gradient_list:
        flat_grads.append(
            np.concatenate([vec.numpy().flatten() if type(vec) != np.ndarray else vec.flatten() for vec in grad]))

    gradient = np.array(flat_grads)
    params = np.array(flat_weights)

    distance_matrix = np.sum(np.square(params[None, :, :] - params[:, None, :]), axis=-1)

    # get median
    distance_vector = distance_matrix.flatten()

    distance_vector.sort()
    median = 0.5 * (
            distance_vector[int(len(distance_vector) / 2)] + distance_vector[int(len(distance_vector) / 2) - 1])
    h = median / (2 * np.log(num_agents + 1))
    kernel = np.exp(distance_matrix[:, :] * (-1.0 / h))
    kernel_gradient = kernel[:, :, None] * (2.0 / h) * (params[None, :, :] - params[:, None, :])

    svpg_grad = (1.0 / temp) * kernel[:, :, None] * gradient[:, None, :] + kernel_gradient[:, :, :]
    svpg_grad = np.mean(svpg_grad[:, :, :], axis=0)

    return svpg_grad#, shape_list  # Let's just return the gradients and let individually chosen optimizers in the actor handle the update.

def return_collapsed_array(sim_results, key):
    """given the list of simulation results, return the collapsed array
    so that all the batches are concatenated, as a list of tf tensors"""

    #print('Length of state history in return_collapsed_arr is {}'.format(len(sim_results)))
    state_history = [sim_results[ind][key] for ind in range(len(sim_results))]
    state_history_arrs = [np.array(state) for state in state_history]
    state_history = np.vstack(state_history_arrs)

    if key =='state_history':
        print("state history shape is {}".format(state_history.shape))
        state_history_full = state_history.reshape(-1, state_history.shape[-1])  # make it
        state_history_tf = tf.squeeze(tf.stack(state_history_full[:]))
    else:
        print("other history shape is {}".format(state_history.shape))
        state_history_full = state_history.flatten()
        state_history_tf = tf.squeeze(tf.stack(state_history_full[:]))

    return state_history_tf
