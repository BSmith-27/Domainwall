import numpy as np
import sys
sys.path.append("..")
import os
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import gym

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

        actor.add(layers.Dense(self.dim_actions * 2, activation='tanh',
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
   
    def __init__(self, input_dims = [14,2], output_dims=[12], wr = 0.01, lr = 0.01):
        
        self.wr = wr
        self.initializer =  tf.keras.initializers.he_uniform()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.lr = lr
        super(WallNet, self).__init__()
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.RMSprop(lr=lr)
        

    def build_model(self):
        InputImage = layers.Input(shape=(self.input_dims[0],1))
        InputNumeric = layers.Input(shape=(self.input_dims[1]))
        cnet = layers.Dense(512, activation=tf.nn.relu,
                               kernel_initializer=self.initializer, 
                           kernel_regularizer=regularizers.L2(self.wr))(InputImage)
       
        cnet = layers.Dense(512, activation=tf.nn.relu,
                               kernel_initializer=self.initializer,
                           kernel_regularizer=regularizers.L2(self.wr))(cnet)
        cnet = layers.Dropout(0.2)(cnet)
        cnet = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer,
                           kernel_regularizer=regularizers.L2(self.wr))(cnet)
        
        
        cnet = layers.Flatten()(cnet)
        
        cnet = Model(inputs=InputImage, outputs=cnet)
        
        numeric = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer,
                              kernel_regularizer=regularizers.L2(self.wr))(InputNumeric)
        numeric = layers.Dropout(0.2)(numeric)
        numeric = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer,
                              kernel_regularizer=regularizers.L2(self.wr))(numeric)
        numeric = layers.Dropout(0.2)(numeric)
        numeric = layers.Dense(256, activation=tf.nn.relu,
                               kernel_initializer=self.initializer,
                              kernel_regularizer=regularizers.L2(self.wr))(numeric)

        numeric = Model(inputs=InputNumeric, outputs=numeric)

        combined = layers.concatenate([cnet.output, numeric.output])
        
        x = layers.Dense(512,activation=tf.nn.relu, kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.L2(self.wr))(combined)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256,activation=tf.nn.relu, kernel_initializer=self.initializer,
                        kernel_regularizer=regularizers.L2(self.wr))(x)
        combined_network = layers.Dense(self.output_dims[0],activation='linear', 
                                        kernel_initializer=self.initializer)(x)
    
        model = Model(inputs=[cnet.input, numeric.input], outputs=combined_network)

        return model
    
    # define forward pass
    def call(self, inputs):
        prediction = self.model(inputs)[:,:]

        return prediction

class WallEnv(gym.Env):

    def __init__(self, model_file, noise_val = 0.01,
                   state_size=128, num_steps = 30,
                   reward_freq = 'all', desired_wall = None, actions_type = 'one',
                 activation = True):
        '''

        actions_type: 'all' or 'one' or 'two': if 'one', will adjust first and last (position and pulse width) automatically
        without need to learn it. If 'two' it will allow for pulse amplitude and width actions only.
        activation (True or False): whether there is a threshold activation V and PW before a change is made to the wall state
        '''

        super(WallEnv, self).__init__()


        #self.observation_space = spaces.Box(low=-10,high=10,shape=(128,))

        #reward_freq = 'all' or 'end'
        self.model_file = model_file
        self.ynet = WallNet()
        self.ynet.compile()
        self.ynet.model.load_weights(self.model_file)
        self.state_size = state_size
        self.noise_val = noise_val
        self.num_steps = num_steps
        self.reward_freq = reward_freq
        self.state = np.zeros(self.state_size) + np.random.normal(size=self.state_size, scale=self.noise_val)
        self.step_number = 0
        self.activation = activation
        self.local_win_size = 7
        self.offset = 5
        self.dyn_model = self.ynet
        self.actions_type = actions_type
        #if self.actions_type =='one':
        #    self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        #elif self.actions_type=='two':
        #    self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        #else:
        #    self.action_space = spaces.Box(low=np.array([0.1, 0.0, 0.0]), high=np.array([0.9, 1.0, 1.0]), dtype=np.float32)

        if desired_wall is not None:
            self.desired_wall = desired_wall
        else:
            desired_wall = np.zeros(self.state_size)
            desired_wall[:self.state_size//4] = 1.0
            desired_wall[self.state_size//4:self.state_size//2] = 0.75
            desired_wall[self.state_size//2:(self.state_size//4 *3)] = 0.5
            desired_wall[(self.state_size//4 * 3) :] = 0.25
            desired_wall = desired_wall*0.5
            self.desired_wall = desired_wall

    def get_reward(self, my_state):
        if self.reward_freq == 'end':
            if self.step_number<self.num_steps:
                reward = 0
            else:
                reward = -mean_squared_error(my_state,self.desired_wall)
        elif self.reward_freq == 'all':
            reward = -mean_squared_error(my_state,self.desired_wall)
        return reward

    def step(self, action):

        #action has three values, first value is position, second is bias amp. third is pulse width


        if self.actions_type == 'all':
            action[0] = np.clip(action[0], 0.1, 0.9)
            wall_pos = action[0]*128
            action_dyn = action[1:]
        elif self.actions_type=='two':
            wall_pos = 0.0+(1.0/self.num_steps)*self.step_number
            wall_pos = 128*wall_pos*0.8 + 2*self.offset
            action_0 = np.clip(action[0], -1, 1)
            action_1 = np.clip(action[1], 0, 1)
            action_dyn = np.array([action_0, action_1])

        else:
            wall_pos = 0.0+(1.0/self.num_steps)*self.step_number
            #wall_pos = np.clip(wall_pos, 0.1, 0.9)
            wall_pos = 128*wall_pos*0.8 + 2*self.offset
            action_dyn = np.array([action, 0.55])



        #print(action_dyn)

        state_subsection_raw = self.state[int(wall_pos - self.local_win_size): int(wall_pos + self.local_win_size)]
        local_mean = np.mean(state_subsection_raw) #this is not right,  we need to figure out something better later...
        state_subsection = state_subsection_raw - local_mean
        self.step_number+=1

        #print(state_subsection, action_dyn)

        new_wall_subsection = self.dyn_model([state_subsection[None,:,None], action_dyn[None,:]])
        new_state = np.copy(self.state)
        new_state[int(wall_pos - self.local_win_size+1): int(wall_pos + self.local_win_size-1)] = new_wall_subsection + local_mean
        if self.activation:
          if np.abs(action_dyn[0]*action_dyn[1])>=0.20:
            self.state = new_state
          else:
            self.state = self.state
        else:
          self.state = new_state

        new_state = self.state
        reward = self.get_reward(new_state)
        done= False


        if self.step_number == self.num_steps:
            done=True
        info = {}

        return new_state, reward, done, info

    def reset(self):
        new_state = np.zeros(self.state_size) + np.random.normal(size=self.state_size, scale=self.noise_val)
        self.state = new_state
        self.step_number = 0
        reward = 0
        return self.state, reward


