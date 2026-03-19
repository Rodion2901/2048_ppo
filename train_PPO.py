from Game import game_2048
from PPO import PPOAgent
from PPO import Memory
from PPO import checkpoint
from PPO import load_checkpoint
from Game import pretty_print
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/2048_ppo_v1")

episodes = 100
start_episode = 1
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

max_results = []
if start_episode != 1:
    agent = load_checkpoint(agent, start_episode)
for episode in range(start_episode, episodes+1):
    state = game.reset()
    done = False
    total_reward = 0
    while not done:

        action = agent.act(state, memory)
        next_state, reward, done = game.step(action)

        memory.push(reward, done)

        state = next_state
        
        total_reward += reward
    agent.update(memory)
    memory.clear()
    
    reward_history.append(total_reward)
    max_tile = source_max_result(state)
    max_results.append(max_tile)

    writer.add_scalar('Reward/Total', total_reward, episode)
    writer.add_scalar('Max_Tile', max_tile, episode)

    if (max_tile >= 512)or(episode % 200 == 0):
         pretty_print(state)

    reward_history.append(total_reward)
    max_results.append(source_max_result(state))
    if episode % 100 == 0:
            print(
                f"Episode: {episode}, "
                f"максимальный результат: {max(max_results)}, "
                f"lr in optimizer: {agent.optimizer.param_groups[0]['lr']:.5f}"
            )
    if episode % 1000 == 0:
        checkpoint(agent.policy, agent.optimizer, agent.scheduler, episode)
print(max(max_results))
writer.close()