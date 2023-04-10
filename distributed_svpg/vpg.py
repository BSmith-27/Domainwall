import numpy as np
import sys
sys.path.append("..")
import os
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
from sklearn.metrics import mean_squared_error

class NeuralNet_A2C(Model):
    # This defines two distinct models
    # One is an actor, another is the critic (value function estimation)
    # Both are fully connected neural networks

    def __init__(self, input_dim=70, dim_actions=6, num_hidden_nodes_1=128,
                 num_hidden_nodes_2=64,
                 actor_lr=0.001, critic_lr=0.002, lstm = False):
        self.num_hidden_nodes_1 = num_hidden_nodes_1
        self.num_hidden_nodes_2 = num_hidden_nodes_2
        self.input_dim = input_dim
        self.dim_actions = dim_actions
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.initializer = tf.keras.initializers.glorot_uniform(seed = np.random.randint(0,50))
        self.lstm = lstm

        super(NeuralNet_A2C, self).__init__()
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def build_actor(self):
        actor = Sequential()
        actor.add(layers.Input(shape=(self.input_dim)))
        actor.add(layers.Conv1D(64, kernel_size=(7), activation = tf.nn.relu, kernel_initializer=self.initializer))
        actor.add(layers.Conv1D(32, kernel_size=(3), activation = tf.nn.relu, kernel_initializer=self.initializer))
        actor.add(layers.AveragePooling1D(2))
        actor.add(layers.Flatten())
        actor.add(layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.relu,
                               kernel_initializer=self.initializer,
                               name='fc_1'))
        actor.add(layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.relu,
                               kernel_initializer=self.initializer, name='fc_2'))

        actor.add(layers.Dense(self.dim_actions * 2, activation='sigmoid',
                               kernel_initializer=self.initializer,
                               name='output_actions_layer'))

        # here the actor will output a mean, standard deviation from which we will sample.

        return actor

    def build_critic(self):
        # critic neural network
        critic = Sequential()
        critic.add(layers.Input(shape=(self.input_dim)))
        critic.add(layers.Conv1D(64, kernel_size=(7), activation = tf.nn.relu, kernel_initializer=self.initializer))
        critic.add(layers.Conv1D(32, kernel_size=(3), activation = tf.nn.relu, kernel_initializer=self.initializer))
        critic.add(layers.AveragePooling1D(2))
        critic.add(layers.Flatten())
        critic.add(layers.Dense(128, activation=tf.nn.relu,
                                kernel_initializer=self.initializer,
                                name='fc_1'))
        critic.add(layers.Dense(64, activation=tf.nn.relu,
                                kernel_initializer=self.initializer, name='fc_2'))
        critic.add(layers.Dense(1, activation='linear',
                                kernel_initializer=self.initializer,
                                name='output_actions_layer'))

        # critic.compile(loss = self.critic_loss, optimizer=tf.optimizers.Adam(lr=self.critic_lr))
        return critic

    # define forward pass
    def call(self, states):
        actions_output = self.actor(states)*3
        # actions_output[0,1] = tf.nn.softplus(actions_output[0,1]) + 1e-5
        value_estimate = self.critic(states)

        return actions_output, value_estimate


#model for dynamics environment
class WallNet(Model):
   
    def __init__(self, input_dims = [14,2], output_dims=[12]):
        
        self.initializer =  tf.keras.initializers.he_uniform(seed = np.random.randint(0,50))
        self.input_dims = input_dims
        self.output_dims = output_dims
        super(WallNet, self).__init__()
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.RMSprop()

    def build_model(self):
        InputImage = layers.Input(shape=(self.input_dims[0],1))
        InputNumeric = layers.Input(shape=(self.input_dims[1]))
        cnet = layers.Dense(512, activation=tf.nn.relu,
                               kernel_initializer=self.initializer )(InputImage)

        cnet = layers.Dense(512, activation=tf.nn.relu,
                               kernel_initializer=self.initializer )(cnet)
        
        cnet = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer )(cnet)
        
        
        cnet = layers.Flatten()(cnet)
        
        cnet = Model(inputs=InputImage, outputs=cnet)

        numeric = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer )(InputNumeric)

        numeric = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer )(numeric)
        
        numeric = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer )(numeric)

        numeric = Model(inputs=InputNumeric, outputs=numeric)

        combined = layers.concatenate([cnet.output, numeric.output])
        
        x = layers.Dense(512,activation=tf.nn.relu, kernel_initializer=self.initializer)(combined)
        x = layers.Dense(256,activation=tf.nn.relu, kernel_initializer=self.initializer)(x)
        combined_network = layers.Dense(self.output_dims[0],activation='linear', 
                                        kernel_initializer=self.initializer)(x)
    
        model = Model(inputs=[cnet.input, numeric.input], outputs=combined_network)

        return model
    
    # define forward pass
    def call(self, inputs):
        prediction = self.model(inputs)[:,:]

        return prediction

class WallEnv:
  def __init__(self, noise_val = 0.01, 
               state_size=128, max_steps = 200, 
               reward_freq = 'end', desired_wall = None,
               model_file=None, thresh = -0.025):
    
    #reward_freq = 'all' or 'end'
    self.model_file = model_file
    self.ynet = WallNet()
    self.ynet.compile()
    self.ynet.model.load_weights(self.model_file)

    self.state_size = state_size
    self.noise_val = noise_val
    self.max_steps = max_steps
    self.reward_freq = reward_freq
    self.state = np.zeros(self.state_size) + np.random.normal(size=self.state_size, scale=self.noise_val)
    self.step_num = 0
    self.offset = 5
    self.local_win_size = 7
    self.local_state_size = 14
    self.thresh = thresh
    self.done = False

    if desired_wall is not None: 
      self.desired_wall = desired_wall
    else:
      desired_wall = np.zeros(self.state_size)
      desired_wall[:self.state_size//2] = 0 
      desired_wall[self.state_size//2 :] = 0.5
      self.desired_wall = desired_wall
    
  def reset(self):
    def start_numpy_version(init_steps = 20):
      new_state = np.zeros(self.state_size) + np.random.normal(size=self.state_size, scale=self.noise_val)
      for i in range(init_steps):
          actions = np.random.randn(3)
          actions = np.clip(actions, 0.1, 0.9)
          actions[1] = (actions[1]*2)-1 #voltage distribution shift
          wall_pos = actions[0]*128+self.offset
          state_subsection = new_state[int(wall_pos-self.local_win_size): int(wall_pos+self.local_win_size)]
          model_state = np.array(state_subsection)
          model_state = tf.convert_to_tensor(model_state)
          model_state = model_state[None,:,None]
          model_actions = np.array(actions[1:])[None,:]
          model_actions = tf.convert_to_tensor(model_actions)
          model_input = [model_state, model_actions]
          wall_pred = self.ynet(model_input)
          wall12 = state_subsection[(self.local_win_size - (self.local_win_size-1)): (self.local_win_size + (self.local_win_size-1))]
          new_wall = wall12+wall_pred
          begin = int(wall_pos - (self.local_win_size-1))
          end = int(wall_pos + (self.local_win_size-1))
          new_state[begin:end]=new_wall
      return new_state
    
    def start_tensorflow_version(init_steps, state_size, noise_val, local_win_size, offset):
      print('state size is {}, noise_val is {}'.format(state_size, noise_val))
      new_state = tf.zeros(state_size) + tf.random.normal(shape=state_size, stddev=noise_val)
      #rewrite the function start_numpy_version, in tensorflow instead of numpy
      for _ in range(init_steps):
        actions = tf.random.normal(shape=(3,))
        actions = tf.clip_by_value(actions, 0.1, 0.9)
        actions[1] = (actions[1]*2)-1 #voltage distribution shift
        wall_pos = actions[0]*128+offset
        state_subsection = new_state[int(wall_pos-self.local_win_size): int(wall_pos+self.local_win_size)]
        model_state = state_subsection[None,:,None]
        model_input = [model_state, actions]
        wall_pred = self.ynet(model_input)
        wall12 = state_subsection[(local_win_size - (local_win_size-1)): (local_win_size + (local_win_size-1))]
        new_wall = wall12+wall_pred
        begin = int(wall_pos - (local_win_size-1))
        end = int(wall_pos + (local_win_size-1))
        new_state[begin:end]=new_wall      
      return new_state
    
    init_steps = 20
    new_state = start_tensorflow_version(init_steps, self.state_size, self.noise_val, self.local_win_size, self.offset)
    self.state = new_state
    reward = self.get_reward(new_state)
    self.step_num = 0
    self.done = False
    return self.state, reward

  def step(self, action, verbose=False):
   
    if verbose: print("ACTIONS: ",action)
    #action has three values, first value is position, second is bias amp. third is pulse width
    #THe first one is used to determine the sub-section of the state
    wall_pos = action[0]*128 + self.offset
    if verbose:
      print("FULL WALL: ", self.state)
      print("EXACT WALL POSITION: ", int(wall_pos))
    self.state_subsection = self.state[int(wall_pos - self.local_win_size): int(wall_pos + self.local_win_size)]
    if verbose: print("LOCAL WALL: ", self.state_subsection)
    
    model_state= np.array(self.state_subsection)
    model_state = tf.convert_to_tensor(model_state)
    model_state = model_state[None,:,None] 
    model_actions= np.array(action[1:])
    model_actions = model_actions[None,:] 
    model_actions = tf.convert_to_tensor(model_actions) 
    if verbose:
      print('model_state is {} and model_actions is {}'.format(model_state, model_actions))

    model_input = [model_state, model_actions]
    wall_pred = self.ynet(model_input)
    wall12 = self.state_subsection[(self.local_win_size - (self.local_win_size-1)): (self.local_win_size + (self.local_win_size-1))]
    new_wall = wall12 + wall_pred
    new_state = np.copy(self.state)
    begin = int(wall_pos - (self.local_win_size-1))
    end =   int(wall_pos + (self.local_win_size-1))
    
    new_state[begin: end] = new_wall
    info = {}
    self.state = new_state
    self.step_num+=1
    cur_reward = -mean_squared_error(new_state,self.desired_wall)
    if self.step_num>=self.max_steps: self.done = True 
    else: self.done = False
    if cur_reward>= self.thresh and self.step_num<30: #try to solve within 20 iterations
       print('reward is {}'.format(cur_reward))
       self.done = True
       reward = cur_reward
       info = {'reward':cur_reward}
  
    done = self.done
    reward = self.get_reward(new_state)

    #print('step number is {} and reward is {}'.format(self.step_num, cur_reward))
    
    return new_state, reward, done, info

  def get_reward(self, my_state):
    #write mse reward for self.desired_state and my_state
    if self.reward_freq == 'end':
      if self.done == False:
        reward = -1.0/ self.max_steps
      else:
        reward = -mean_squared_error(my_state,self.desired_wall)
        
    elif self.reward_freq == 'all':
      reward = -mean_squared_error(my_state,self.desired_wall)
      
    return reward
