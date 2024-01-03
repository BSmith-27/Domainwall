from .vpg import NeuralNet_A2C, WallEnv
from .utils import reshape_gradient,\
    SteinUpdateStep, return_collapsed_array, return_targets, generalized_advantage_estimate
from mpi4py import MPI
import sys
from os import path
import tensorflow as tf
import numpy as np
import gym
import tensorflow_probability as tfp
import time
import os

sys.path.append("..")

tf.get_logger().setLevel('ERROR')  # Suppress tf warnings

# Communications setup
comm_world = MPI.COMM_WORLD
global world_rank, world_size, world_rank_hvd
world_rank = comm_world.Get_rank()
world_size = comm_world.Get_size()
print("world rank is {} and world size is {}".format(world_rank, world_size))

num_agents_per_gpu = 6
#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_visible_devices(gpu_devices[world_rank // num_agents_per_gpu], 'GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[world_rank // num_agents_per_gpu], True)

#The policy is a global variable. There will be one policy per MPI process

policy = NeuralNet_A2C(input_dim=(128,1), dim_actions=1, actor_lr = 0.001, critic_lr = 0.0013)
model_name = r'trained_surrogate_new_weights.h5'
model_path = os.path.join(os.path.abspath(os.getcwd()), 'distributed_svpg/')
model_file = os.path.join(model_path,model_name)
INS_Env = WallEnv(model_file = model_file)

def print_rank(*args, **kwargs):
    if world_rank == 0:
        print(*args, **kwargs)

def getLocalPolicyGrad(sim_results, verbose = False, svpg = False, gamma=0.99):
    # Concatenate list of dictionaries
    # Calculate the advantage estimate
    # return the gradient

    state_history_tf = return_collapsed_array(sim_results, 'state_history')
    actions_history_tf = return_collapsed_array(sim_results, 'actions_history')
    reward_history_tf = return_collapsed_array(sim_results, 'reward_history')
    done_history_tf = return_collapsed_array(sim_results, 'done_history')

    #print(state_history_tf.shape, actions_history_tf.shape,
    #      reward_history_tf.shape, done_history_tf.shape)
    state_history_tf = state_history_tf[:,:,None]
    _, old_state_values = policy(state_history_tf)
    _, new_state_values = policy(state_history_tf[1:,:,:])

    advantage_estimate,value_target = generalized_advantage_estimate(gamma=gamma, lamda=0.98, 
    value_old_state=old_state_values, value_new_state=new_state_values, 
    reward=reward_history_tf, done=done_history_tf)
    
    #Code below is sometimes useful for normalizing
    #advantage_estimate = (advantage_estimate - tf.math.reduce_mean(advantage_estimate)) / tf.math.maximum(tf.math.reduce_std(advantage_estimate), tf.constant(1.0E-6))

    #Policy gradient
    dim_actions = policy.dim_actions
    with tf.GradientTape() as tape:
        # Calculate the policy gradient
        actions_mean_tt = tf.reshape(policy.actor(state_history_tf), (-1, dim_actions, 2))

        #print('actions mean shape is {}, actions history is {}'.format(actions_mean_tt.shape,
        #                                                               actions_history_tf.shape))

        lognorm_dist = tfp.distributions.MultivariateNormalDiag(
            tf.nn.softplus(tf.reshape(actions_mean_tt, (done_history_tf.shape[0], dim_actions, 2))[:, :, 0]),
            tf.nn.softplus(tf.reshape(actions_mean_tt, (done_history_tf.shape[0], dim_actions, 2))[:, :, 1])).log_prob(
            tf.reshape(actions_history_tf,(done_history_tf.shape[0],dim_actions)))

        loss = -tf.reduce_mean(lognorm_dist * advantage_estimate)  # gradient of objective function
        gradients = tape.gradient(loss, policy.actor.trainable_variables)

    if verbose:
        print_rank('Actor loss is {}'.format(np.round(loss,5)))
    if svpg==False:
        policy.actor.optimizer.apply_gradients(zip(gradients, policy.actor.trainable_variables))

    # Critic loss
    for _ in range (5):
        with tf.GradientTape() as tape:
            _, state_values = policy(state_history_tf)
            state_values = state_values[:,0] * done_history_tf
            td_error = tf.reduce_mean((np.squeeze(np.array(value_target)) - state_values) ** 2)  # TD_error
            valueGradient = tape.gradient(td_error, policy.critic.trainable_variables)
        policy.critic.optimizer.apply_gradients(zip(valueGradient, policy.critic.trainable_variables))

    if verbose:
        print_rank('Critic loss is {}'.format(np.round(td_error, 2)))

    return gradients

def run_simulation(numSimRuns = 10, agent_comms=[0,0], verbose = False):

    state_history = []
    actions_history = []
    reward_history = []
    done_history = []
    episode_history = []
    px,py = agent_comms
    #gym.make('BipedalWalker-v3')

    for _ in range(numSimRuns):
        state,_ = INS_Env.reset()  
        reward_total = 0
        done = False
     
        if verbose: 
            print_rank('state shape is {}'.format(state.shape))
        while not done:
            #print('epiode #{} with step #{}'.format(ep_num, steps))
            # Take an action
            #print('state is {}'.format(state))

            actions_output, _ = policy(state[None,:, None])

            actions_output = tf.reshape(actions_output[0, :], (-1, 2))

            # Sample the policy to get the action
            output_action = tfp.distributions.MultivariateNormalDiag(
                tf.nn.softplus(actions_output[:, 0]), tf.nn.softplus(actions_output[:, 1])).sample(1)

            output_actions = np.squeeze(tf.clip_by_value(output_action, 0.1, 0.9).numpy())
            if verbose:
                print_rank("output actions is {}".format(output_actions))
                print_rank("step number of episode is {}".format(INS_Env.step_num))
            
            # Take the selected action in the environment
            next_state, reward, done, info = INS_Env.step(output_actions)
            if len(info.keys())>0:
                print('reward > thresh')
                filepath_actor = 'actor_' + str(px) + str(py) + \
                '_reward=' + str(np.round(info['reward'],4)) +'.h5'

                filepath_critic = 'critic_' + str(px) + str(py) + \
                '_reward=' + str(np.round(info['reward'],4)) +'.h5'
                
                print('saving results in file {}'.format(filepath_actor))
                save_folder_name = 'intermediate_break_results/'
                if not os.path.exists(save_folder_name):
                    try: 
                        os.mkdir(save_folder_name)
                        print('Making directory {}'.format(save_folder_name))
                    except: 
                        pass

                policy.actor.save_weights(path.join(save_folder_name, filepath_actor))
                policy.critic.save_weights(path.join(save_folder_name, filepath_critic))

            actual_reward = reward #if done else 0.0
            reward_total += actual_reward

            next_state = np.squeeze(next_state)

            # Save results
            actions_history.append(output_action)
            state_history.append(state.flatten())
            reward_history.append(reward)
            done_history.append(np.array(0.0, dtype=np.float32) if done else np.array(1.0, dtype=np.float32))

            # First up, the actor network
            state = next_state.flatten()
            
        episode_history.append(reward_total)

    state_history_tf = tf.squeeze(tf.stack(state_history[:]))
    actions_history_tf = tf.squeeze(tf.stack(actions_history[:]), axis=1)
    reward_history_tf = tf.squeeze(tf.stack(reward_history[:]))
    done_history_tf = tf.squeeze(tf.stack(done_history[:]))

    results = {'state_history': state_history_tf,
               'actions_history': actions_history_tf,
               'reward_history': reward_history_tf, 'done_history': done_history_tf,
               'episode_history': np.squeeze(np.array(episode_history))}

    return results

def train_svpg_MPI(iterations=800, batch_size=1, numSimRuns = 1, svpg = True,
                   gamma = 0.90, stein_temp = 2.0, save_folder_name = 'svpg_results/'):

    #Setup of the communications
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_agents = size / batch_size
    comm2D = comm.Create_cart([num_agents, batch_size])
    commX = comm2D.Sub(remain_dims=[False, True])
    commY = comm2D.Sub(remain_dims=[True, False])
    px, py = comm2D.Get_coords(rank)

    #Get local policy weights
    localWeight = policy.actor.get_weights()
    localCriticWeight = policy.critic.get_weights()

    #Broadcast so each copy of the root agent has the same weights
    localWeight = commX.bcast(localWeight, root=0)
    localCriticWeight = commX.bcast(localCriticWeight, root=0)

    #Set the provided weights
    policy.actor.set_weights(localWeight)
    policy.critic.set_weights(localCriticWeight)

    rewards_list = []
    t0 = time.time()
   
    for iteration in range(iterations):
        
        sim_result_i = run_simulation(numSimRuns=numSimRuns, agent_comms = [px, py])
        sim_results = commX.gather(sim_result_i, root=0)

        if py == 0: #each master agent
            #Get the rewards
            rewards_arr = [sim_results[ind]['episode_history'] for ind in range(len(sim_results))]
            rewards_arr = np.array(rewards_arr).flatten()

            #Calculate the policy gradient for each agent. If svpg is False, then gradient is applied
            # Critic loss is calculated as well and applied, regardless of svpg status.

            localGrad = getLocalPolicyGrad(sim_results, svpg = svpg, gamma=gamma)
            #print('localgrad is {}'.format(localGrad))
            #If doing SVPG
            if svpg:
                # Gather them all up - all gradients and all weights for SVPG step
                allGrads = commY.allgather(localGrad)
                localWeight = policy.actor.get_weights()
                allWeights = commY.allgather(localWeight)

                #Stein Update step
                updatedGradient = SteinUpdateStep(allWeights, allGrads, num_agents=num_agents, temp=stein_temp)
                updatedGradient_reshaped = reshape_gradient(updatedGradient[px], localWeight)

                #Apply the Stein Update step to each actor
                policy.actor.optimizer.apply_gradients(zip(updatedGradient_reshaped, policy.actor.trainable_variables))

            localWeight = policy.actor.get_weights()
            localCriticWeight = policy.critic.get_weights()

            #get the rewards
            rwd_sum = np.squeeze(np.sum(rewards_arr))/(batch_size*numSimRuns)
            all_rewards = commY.allgather(rwd_sum)
            rewards_list.append(np.array(all_rewards).flatten())

        #Get weights of actor and critic from root agent, broadcast alogn the row
        localWeight = commX.bcast(localWeight, root=0)
        localCriticWeight = commX.bcast(localCriticWeight, root=0)

        #Set local weights to match the updated ones
        policy.actor.set_weights(localWeight)
        policy.critic.set_weights(localCriticWeight)

        if py==0 and iteration%5==0:
            print_rank('Completed iteration {} with mean reward {:.4f} and max {:.4f}'.format(iteration,
                                                      np.mean(all_rewards),np.max(all_rewards)))
            
            #print_rank('Completed iteration {}, agent rewards were {}'.format(iteration, all_rewards))

        # for saving, need to only save the roots
        '''
        if iteration%10==0 and py==0 :
            filepath_actor = 'Actor_' +str(px) + '_ep_' + str(iteration) + '.h5'
            filepath_critic = 'Critic_' + str(px) + '_ep_' + str(iteration) + '.h5'

            #policy.actor.save_weights(path.join(save_folder_name, filepath_actor))
            #policy.critic.save_weights(path.join(save_folder_name, filepath_critic))
            if px==0:
                np.save(path.join(save_folder_name, filename_rewards), np.array(rewards_list))
        '''

    t1 = time.time()
    print_rank('It took {:.2f}s to complete all iterations'.format(t1-t0))
    if py==0:
        filepath_actor = 'Actor_' +str(px) + '_ep_' + str(iteration) + '.h5'
        filepath_critic = 'Critic_' + str(px) + '_ep_' + str(iteration) + '.h5'

        policy.actor.save_weights(path.join(save_folder_name, filepath_actor))
        policy.critic.save_weights(path.join(save_folder_name, filepath_critic))
        if px==0:
            filename_rewards = 'training_history.npy'
            np.save(path.join(save_folder_name, filename_rewards), np.array(rewards_list))
    return

