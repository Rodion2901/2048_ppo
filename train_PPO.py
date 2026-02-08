from Game import game_2048
from Game import pretty_print
from PPO import PPOAgent
from PPO import Memory
import numpy as np
episodes = 300

game = game_2048()
state_dim = game.map
action_dim = 4
agent = PPOAgent(state_dim, action_dim)
memory = Memory()
reward_history = []
update_timestep = 2000
time_step = 0
for episode in range(1, episodes+1):
    state = game.reset()
    done = False
    total_reward = 0
    while not done:
        time_step += 1
        action, log_probs = agent.act(state)
        next_state, reward, done = game.step(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_probs)
        memory.rewards.append(reward)
        memory.dones.append(done)
        if time_step % update_timestep == 0:
            agent.update(memory)
            memory.clear()
        state = next_state
        
        total_reward += reward
        if ((episode % 50 == 0 ) or (episode>200))and (done):
             print(f"Episode: {episode}")
             pretty_print(state)
    reward_history.append(total_reward)
    if episode % 100 == 0:
            success_rate = np.mean([1 if r > 0 else 0 for r in reward_history[-100:]])
            print(
                f"Episode: {episode}, "
                f"Success Rate: {success_rate:.2%}, "
            )


    
