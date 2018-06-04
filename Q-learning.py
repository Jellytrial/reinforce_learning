@@ -0,0 +1,101 @@
import numpy as np
import pandas as pd
import  time #控制探索者移动速度

np.random.seed(2) #计算机产生一组伪随机数列

N_STATES = 6 #开始距离宝藏的距离
ACTIONS = ['left','right'] #探索者可选动作
EPSILON = 0.9 #greedy police
ALPHA = 0.1 #learning rate 学习效率
LAMBDA = 0.9 #discount factor 未来奖励衰减值
MAX_EPISODES = 13 #训练回合数
FRESH_TIME = 0.1 #走一步花的时间长度

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  #用pandas创建表格，初始都为0
        columns=actions,
    )
    #print(table)   #显示表格
    return table

def choose_action(state, q_table):
    #选择动作
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform( ) > EPSILON) or (state_actions.all( ) == 0): #非贪婪或这个状态还没有被探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax( ) #贪婪模式
    return action_name


def get_env_feedback(S, A):
    #如何与环境进行反馈
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S+1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S-1
    return S_,R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 0   # 回合初始位置，最左边
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_,R= get_env_feedback(S, A)  # 实施行为并得到环境的反馈
            q_predict = q_table.ix[S, A]    # 估算的(状态-行为)值
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode 断开while循环，回到for循环进入下一个回合

            q_table.ix[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table #在训练完之后看q table的值的样子


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)




