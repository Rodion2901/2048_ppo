from Game import game_2048
from PPO import PPOAgent
from PPO import Memory
from PPO import checkpoint
from PPO import load_checkpoint
from Game import pretty_print

episodes = 20000
start_episode = 10000
def source_max_result(map):
    a = 0
    for i in map:
         for j in range(4):
              if a < i[j]:
                   a = i[j]
    return a
game = game_2048()
state_dim = game.map
action_dim = 4
agent = PPOAgent(state_dim, action_dim)
memory = Memory()
reward_history = []
update_timestep = 2000
time_step = 0
max_results = []
if start_episode != 1:
    agent = load_checkpoint(agent, start_episode)
for episode in range(start_episode, episodes+1):
    state = game.reset()
    done = False
    total_reward = 0
    while not done:
        if episode % 500 == 0:
             pretty_print(state)
        time_step += 1
        action = agent.act(state, memory)
        next_state, reward, done = game.step(action)

        memory.push(reward, done)

        if time_step % update_timestep == 0:
            agent.update(memory)
            memory.clear()
        state = next_state
        
        total_reward += reward
        if episode % 500 == 0:
             pretty_print(state)
    reward_history.append(total_reward)
    max_results.append(source_max_result(state))
    if episode % 100 == 0:
            print(
                f"Episode: {episode}, "
                f"максимальный результат: {max(max_results)}"
            )
    if episode % 500 == 0:
        checkpoint(agent.policy, agent.optimizer, episode)
print(max(max_results))