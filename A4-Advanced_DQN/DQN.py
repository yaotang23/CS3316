import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
class Duelnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Duelnet, self).__init__()  # 调用父类的构造函数
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # Shared Part
        self.A_head = torch.nn.Linear(hidden_dim, action_dim)
        self.V_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.A_head(F.relu(self.fc1(x)))
        V = self.V_head(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device,mode='vanilla'):
        self.action_dim = action_dim
        
        self.mode = mode
        if mode=='dueling' or mode== 'double+dueling':
            self.target_q_net = Duelnet(state_dim, hidden_dim,action_dim).to(device)
            self.q_net = Duelnet(state_dim, hidden_dim,action_dim).to(device)
        else:
            self.target_q_net = Qnet(state_dim, hidden_dim,
                                    self.action_dim).to(device)
            self.q_net = Qnet(state_dim, hidden_dim,
                            self.action_dim).to(device)  
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        #print(states.shape, actions.shape, rewards.shape, next_states.shape,)
        if self.mode=='double'or self.mode=='double+dueling':
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_values = self.q_net(states).gather(1, actions)  # Q(s,a)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.optimizer.zero_grad()  
        dqn_loss.backward() 
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1
mode_list = ['vanilla','double','dueling','double+dueling']
Return_lists = dict({m:None for m in mode_list})
Average_return_lists = dict({m:None for m in mode_list})
Eval_reward = dict({m:0 for m in mode_list})
Episodes_list = []
for mode_name in mode_list:
    print("Now we starts to train {} mode".format(mode_name))
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 3
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 5000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device,mode=mode_name)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state,_ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, info= env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    Return_lists[mode_name] = return_list
    Episodes_list = episodes_list
    
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('{}DQN on {}'.format(mode_name,env_name))
    fig_name = mode_name + 'DQN'+'_'+str(num_episodes)+'_'+str(hidden_dim)
    plt.savefig(fig_name)
    plt.clf()

    mv_return = moving_average(return_list, 9)
    Average_return_lists[mode_name]=mv_return
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('{} DQN on {}'.format(mode_name,env_name))
    fig_name = 'Average ' +fig_name
    plt.savefig(fig_name)
    plt.clf()
    '''
    #Evaluate the agent
    env2 = gym.make(env_name,render_mode ='human')
    total_reward = 0
    for i in range(20):
        #step = 0
        epi_reward = 0
        state,_ = env2.reset()
        env2.render()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, truncated,_ = env2.step(action)
            state = next_state
            epi_reward += reward
            total_reward += reward
            #step +=1
            #print("step: ",step)
    env2.close()
    Eval_reward[mode_name] = total_reward//20
    print("Eval ",mode_name," gets totol_reward: ",total_reward)
    
print('Eval_reward: ',Eval_reward)
'''
for mode_name in mode_list:
    plt.plot(Episodes_list,Return_lists[mode_name], label=mode_name)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.ylim(-500, 0)
plt.xlim(0, num_episodes)
plt.title('Performance of DQN and its variants in MountainCar-v0')
plt.legend()
plt.savefig("all"+'_'+str(num_episodes)+'_'+str(hidden_dim))
plt.clf()
'''
print('Eval_reward: ',Eval_reward)
'''
for mode_name in mode_list:
    plt.plot(Episodes_list,Average_return_lists[mode_name], label=mode_name)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.ylim(-2000, 0)
plt.xlim(0, num_episodes)
plt.title('Average performance of DQN and its variants in MountainCar-v0')
plt.legend()
plt.savefig("average-all"+'_'+str(num_episodes)+'_'+str(hidden_dim))