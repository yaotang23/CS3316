import numpy as np
import gymnasium as gym

class CliffWalking(object):
    def __init__(self, grid_height,grid_width):
        self.height = grid_height
        self.width = grid_width
        self.start_state = (self.height-1,0) # the start state is a tuple
        self.state = self.start_state # the current state is a tuple
        self.cliff = [(self.height-1,i) for i in range(1,self.width-1)] # the cliff is a list of tuples
        self.goal_state = (self.height-1,self.width-1) # the goal state is a list of tuples
        
    def step(self,action):
        # action 0,1,2,3: up,right,down,left
        '''
        if action==1 and self.state[1]+1 < self.width:
            self.state = (self.state[0],self.state[1]+1)
        if action==3 and self.state[1]-1 >= 0:
            self.state = (self.state[0],self.state[1]-1)
        if action==0 and self.state[0]-1 >= 0:
            self.state = (self.state[0]-1,self.state[1])
        if action==2 and self.state[0]+1 < self.height:
            self.state = (self.state[0]+1,self.state[1])
            
        if self.state in self.cliff:
            done = True
            reward = -100.0
        else:
            reward = -1.0
        if self.state == self.goal_state:
            done = True
            
        return self.state,reward,done
        '''
        reward = -1.0
        x = -5.0
        if action==1 :
            if self.state[1]+1 < self.width:
                self.state = (self.state[0],self.state[1]+1)
            else:
                reward = x
        if action==3 :
            if self.state[1]-1 >= 0:
                self.state = (self.state[0],self.state[1]-1)
            else:
                reward = x
        if action==0 :
            if self.state[0]-1 >= 0:
                self.state = (self.state[0]-1,self.state[1])
            else:
                reward = x
        if action==2 :
            if self.state[0]+1 < self.height:
                self.state = (self.state[0]+1,self.state[1])
            else:
                reward = x
        done = False
        if self.state in self.cliff:
            done = True
            reward = -100.0
        if self.state == self.goal_state:
            done = True
        return self.state,reward,done
    
    def reset(self):
        self.state = self.start_state
        return self.state

def epsilon_greedy(epsilon,Q,state):
    if np.random.rand() < epsilon:
        action = np.random.randint(0,4)
    else:
        action = np.argmax(Q[state[0],state[1],:])
    return action

def main():
    '''Sarsa for on-policy control'''
    #Initialize Q(s,a) arbitrarily using gaussian distribution
    height = 4
    width = 12
    epsilon = 0.3
    alpha = 0.1
    gamma = 1.0
    num_episodes = 50000
    Q = np.random.normal(0, 0.1, (height, width, 4))
    Q [0,1:width-1,:] = 0.0
    env = CliffWalking(height,width)# Repeat for each episode
    for i in range(num_episodes):
        done = False
        state = env.reset()
        action = epsilon_greedy(epsilon,Q,state)
        while not done:
            next_state,reward,done = env.step(action)
            next_action = epsilon_greedy(epsilon,Q,state)
            Q[state[0],state[1],action] = Q[state[0],state[1],action] + alpha*(reward + gamma*Q[next_state[0],next_state[1],next_action] - Q[state[0],state[1],action])
            state = next_state
            action = next_action
            

    '''Q-learning for off-policy control'''
    Q_2 = np.random.normal(0, 0.1, (height, width, 4))
    Q_2 [0,1:width-1,:] = 0.0
    for i in range(num_episodes):
        done = False
        state = env.reset()
        action = epsilon_greedy(epsilon,Q_2,state)
        while not done:
            next_state,reward,done = env.step(action)
            next_action = epsilon_greedy(epsilon,Q_2,state)
            #the difference between Q-learning and Sarsa is that the action for update is the action with the max Q value
            next_action_for_update = np.argmax(Q_2[next_state[0],next_state[1],:])
            Q_2[state[0],state[1],action] = Q_2[state[0],state[1],action] + alpha*(reward + gamma*Q_2[next_state[0],next_state[1],next_action_for_update] - Q_2[state[0],state[1],action])
            state = next_state
            action = next_action
            
    # Evaluation
    gym_env = gym.make('CliffWalking-v0',render_mode="human")
    # Evaluate the SARSA result
    obs, info = gym_env.reset()
    state = (int(obs/12),obs%12)
    done = False
    truncated = False
    while not done and not truncated:
        action = np.argmax(Q[state[0],state[1],:])
        print(action)
        obs, reward, done, truncated, info = gym_env.step(action)
        state=(int(obs/12),obs%12)
    gym_env.close()
    
    # Evaluate the Q-Learning result
    obs, info = gym_env.reset()
    state = (int(obs/12),obs%12)
    done = False
    truncated = False
    while not done and not truncated:
        action = np.argmax(Q_2[state[0],state[1],:])
        obs, reward, done, truncated, info = gym_env.step(action)
        state=(int(obs/12),obs%12)
    gym_env.close()
    
if __name__=="__main__":
    main()