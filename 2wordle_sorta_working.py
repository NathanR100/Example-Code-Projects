# -*- coding: utf-8 -*-
"""2wordle_sorta_working.ipynb

import pandas as pd
import random
import numpy as np

# Load the word list from a CSV file
wordle = pd.read_csv("wordle.csv")
word_list = wordle["word"].to_list()

# Filter out any words that are not exactly five letters
word_list = [word for word in word_list if len(word) == 5 and word.isalpha()]



# import numpy as np
import random

class WordleEnvironment:
    def __init__(self, word_list):
        self.word_list = word_list  # A list of possible target words
        self.target_word = None
        self.attempts_remaining = 6
        self.state = None

    def reset(self):
        self.target_word = random.choice(self.word_list)  # Select a random word as the target
        self.attempts_remaining = 6
        self.state = [['_' for _ in range(5)] for _ in range(6)]  # Reset state: 6 attempts, 5 letters each
        return self.state

    def step(self, guess):
        if len(guess) != 5 or not guess.isalpha():
            raise ValueError("Guess must be a five-letter word.")

        feedback = self.provide_feedback(guess)
        self.update_state(guess, feedback)
        self.attempts_remaining -= 1
        done = guess == self.target_word or self.attempts_remaining == 0
        reward = self.calculate_reward(feedback)
        return self.state, reward, done

    def provide_feedback(self, guess):
        feedback = []
        for i, char in enumerate(guess):
            if char == self.target_word[i]:
                feedback.append('green')
            elif char in self.target_word:
                feedback.append('yellow')
            else:
                feedback.append('black')
        return feedback

    def update_state(self, guess, feedback):
        attempt_index = 6 - self.attempts_remaining
        self.state[attempt_index] = feedback  # Update the state with feedback

    def calculate_reward(self, feedback):
        return sum(1 if f == 'green' else 0.5 if f == 'yellow' else 0 for f in feedback)  # Simple reward function

class WordleAgent:
    def __init__(self, word_list):
        self.word_list = word_list  # List of all possible words the agent can guess

    def guess(self):
        # Initially, just randomly select a word from the possible list
        return random.choice(self.word_list)

    # Later we will add methods here to update the agent's strategy based on feedback



class QLearningWordleAgent(WordleAgent):
    def __init__(self, word_list, epsilon=0.1, learning_rate=0.1, discount_factor=0.9, decay_rate=0.99):
        super().__init__(word_list)
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate
        self.q_table = {}

    def update_epsilon(self):
        # Reduce epsilon over time to shift from exploration to exploitation
        self.epsilon *= self.decay_rate
        self.epsilon = max(self.epsilon, 0.01)  # Ensure epsilon does not go below a certain threshold

    def learn(self, state, action, reward, next_state):
        current_state_str = self.state_to_string(state)
        next_state_str = self.state_to_string(next_state)
        self.q_table.setdefault((current_state_str, action), 0)  # Default Q-value initialization

        max_future_q = max(self.q_table.get((next_state_str, a), 0) for a in self.word_list)
        old_q_value = self.q_table[(current_state_str, action)]
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q_value)
        self.q_table[(current_state_str, action)] = new_q_value

        print(f"Updated Q-value for ({current_state_str}, {action}): {new_q_value}")  # Debug output

    def run_episode(self):
        self.update_epsilon()  # Update epsilon at the start of each episode
        # Rest of the episode execution logic


    def state_to_string(self, state):
        # Convert the state from a list of lists to a single string
        return ''.join([''.join(row) for row in state])

    def guess(self, state):
        # Use the state_to_string method to convert the current state to a string
        state_str = self.state_to_string(state)

        # Implement epsilon-greedy strategy here
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action
            return random.choice(self.word_list)
        else:
            # Exploitation: Choose the best action based on the current Q-table
            return self.choose_best_action(state_str)

    def choose_best_action(self, state_str):
        # Find the action with the highest Q-value for the current state
        best_action = None
        max_q_value = float('-inf')

        for action in self.word_list:
            q_value = self.q_table.get((state_str, action), 0)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

        return best_action if best_action else random.choice(self.word_list)

import random
import numpy as np


class EnhancedWordleEnvironment(WordleEnvironment):
    def calculate_reward(self, feedback):
        # More nuanced reward calculation
        green_count = feedback.count('green')
        yellow_count = feedback.count('yellow')
        black_count = feedback.count('black')
        return 2 * green_count + yellow_count - 0.5 * black_count

    def encode_state(self):
        # Compact state encoding using string representation
        return ''.join([''.join(feed) for feed in self.state])

class ImprovedQLearningWordleAgent(QLearningWordleAgent):
    def update_epsilon(self):
        # Epsilon decays each episode to minimize exploration over time
        self.epsilon = max(self.epsilon * self.decay_rate, 0.01)

    def guess(self, state):
        state_str = self.state_to_string(state)
        if np.random.rand() < self.epsilon:
            return random.choice(self.word_list)
        else:
            return self.choose_best_action(state_str)

    def choose_best_action(self, state_str):
        # Chooses the best action from Q-table or random if no known good actions
        possible_actions = self.q_table.get(state_str, {})
        if not possible_actions:
            return random.choice(self.word_list)
        return max(possible_actions, key=possible_actions.get, default=random.choice(self.word_list))

    def state_to_string(self, state):
        # Convert state from a list of lists to a single string
        return ''.join([''.join(feed) for feed in state])

def run_simulation(agent, environment, episodes, report_every=100):
    total_rewards = []
    successes = []

    for episode in range(episodes):
        state = environment.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.guess(state)
            next_state, reward, done = environment.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        successes.append(1 if action == environment.target_word else 0)

        if (episode + 1) % report_every == 0:
            print(f"Episode {episode + 1}: Avg Reward = {np.mean(total_rewards[-report_every:]):.2f}, Success = {np.mean(successes[-report_every:])*100:.2f}%")

    # Plot performance metrics
    plt.figure(figsize=(10, 4))
    plt.plot(np.convolve(total_rewards, np.ones(report_every)/report_every, mode='valid'), label='Smoothed Total Reward')
    plt.plot(np.convolve(successes, np.ones(report_every)/report_every, mode='valid'), label='Smoothed Success Rate')
    plt.title('Agent Performance Over Time')
    plt.xlabel('Episodes')
    plt.ylabel('Performance')
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt

def run_simulation(agent, environment, episodes, report_every=100):
    total_rewards = []
    success_counts = []
    move_counts = []

    for episode in range(episodes):
        state = environment.reset()
        done = False
        total_reward = 0
        moves = 0

        while not done:
            action = agent.guess(state)
            next_state, reward, done = environment.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            total_reward += reward
            moves += 1
            if done and environment.target_word == action:
                success_counts.append(1)
            elif done:
                success_counts.append(0)

        total_rewards.append(total_reward)
        move_counts.append(moves)

        agent.update_epsilon()  # Update epsilon after each episode

        if (episode + 1) % report_every == 0:
            print(f"Episode {episode + 1}: Average Reward = {np.mean(total_rewards[-report_every:]):.2f}, Success Rate = {np.mean(success_counts[-report_every:])*100:.2f}%, Average Moves = {np.mean(move_counts[-report_every:]):.2f}")

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(total_rewards)
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 3, 2)
    plt.plot(np.cumsum(success_counts) / np.arange(1, episodes + 1))
    plt.title('Cumulative Success Rate Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.subplot(1, 3, 3)
    plt.plot(move_counts)
    plt.title('Moves Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')

    plt.tight_layout()
    plt.show()

# Example of setting up and running the simulation
environment = EnhancedWordleEnvironment(word_list)
agent = ImprovedQLearningWordleAgent(word_list, epsilon=0.9, learning_rate=0.1, discount_factor=0.9, decay_rate=0.99)
run_simulation(agent, environment, episodes=1000, report_every=100)

import matplotlib.pyplot as plt

def run_simulation(agent, environment, episodes, report_every=100):
    total_rewards = []
    success_counts = []
    move_counts = []

    for episode in range(episodes):
        state = environment.reset()
        done = False
        total_reward = 0
        moves = 0

        while not done:
            action = agent.guess(state)
            next_state, reward, done = environment.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

            total_reward += reward
            moves += 1
            if done and environment.target_word == action:
                success_counts.append(1)
            elif done:
                success_counts.append(0)

        total_rewards.append(total_reward)
        move_counts.append(moves)

        agent.update_epsilon()  # Update epsilon after each episode

        if (episode + 1) % report_every == 0:
            print(f"Episode {episode + 1}: Average Reward = {np.mean(total_rewards[-report_every:]):.2f}, Success Rate = {np.mean(success_counts[-report_every:])*100:.2f}%, Average Moves = {np.mean(move_counts[-report_every:]):.2f}")

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(total_rewards)
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 3, 2)
    plt.plot(np.cumsum(success_counts) / np.arange(1, episodes + 1))
    plt.title('Cumulative Success Rate Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.subplot(1, 3, 3)
    plt.plot(move_counts)
    plt.title('Moves Per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Number of Moves')

    plt.tight_layout()
    plt.show()

# Example of setting up and running the simulation
environment = EnhancedWordleEnvironment(word_list)
agent = ImprovedQLearningWordleAgent(word_list, epsilon=0.9, learning_rate=0.1, discount_factor=0.9, decay_rate=0.99)
run_simulation(agent, environment, episodes=10000, report_every=100)

# class WordleAgent:
#     def __init__(self, word_list):
#         self.word_list = word_list  # List of all possible words the agent can guess

#     def guess(self):
#         # Initially, just randomly select a word from the possible list
#         return random.choice(self.word_list)

#     # Later we will add methods here to update the agent's strategy based on feedback

# class QLearningWordleAgent(WordleAgent):
#     def __init__(self, word_list):
#         super().__init__(word_list)
#         self.q_table = {}  # Initializes an empty Q-table
#         self.learning_rate = 0.1
#         self.discount_factor = 0.9

#     def learn(self, state, action, reward, next_state):
#         # Convert state to a simple string that can be used as a dictionary key
#         current_state_str = self.state_to_string(state)
#         next_state_str = self.state_to_string(next_state)

#         # Initialize Q values for state-action pairs if not already present
#         if (current_state_str, action) not in self.q_table:
#             self.q_table[(current_state_str, action)] = 0
#         if next_state_str not in self.q_table:
#             max_future_q = 0
#         else:
#             max_future_q = max(self.q_table.get((next_state_str, a), 0) for a in self.word_list)

#         # Q-learning formula to update the Q-value
#         old_q_value = self.q_table[(current_state_str, action)]
#         new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q_value)
#         self.q_table[(current_state_str, action)] = new_q_value

#     def state_to_string(self, state):
#         return ''.join([''.join(row) for row in state])

# import numpy as np

# import numpy as np

# class QLearningWordleAgent(WordleAgent):
#     def __init__(self, word_list, epsilon=0.1, learning_rate=0.1, discount_factor=0.9, decay_rate=0.99):
#         super().__init__(word_list)
#         self.initial_epsilon = epsilon
#         self.epsilon = epsilon
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.decay_rate = decay_rate
#         self.q_table = {}

#     def update_epsilon(self):
#         # Reduce epsilon over time to shift from exploration to exploitation
#         self.epsilon *= self.decay_rate
#         self.epsilon = max(self.epsilon, 0.01)  # Ensure epsilon does not go below a certain threshold

#     def learn(self, state, action, reward, next_state):
#         current_state_str = self.state_to_string(state)
#         next_state_str = self.state_to_string(next_state)
#         self.q_table.setdefault((current_state_str, action), 0)  # Default Q-value initialization

#         max_future_q = max(self.q_table.get((next_state_str, a), 0) for a in self.word_list)
#         old_q_value = self.q_table[(current_state_str, action)]
#         new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q_value)
#         self.q_table[(current_state_str, action)] = new_q_value

#         print(f"Updated Q-value for ({current_state_str}, {action}): {new_q_value}")  # Debug output

#     def run_episode(self):
#         self.update_epsilon()  # Update epsilon at the start of each episode
#         # Rest of the episode execution logic


#     def state_to_string(self, state):
#         # Convert the state from a list of lists to a single string
#         return ''.join([''.join(row) for row in state])

#     def guess(self, state):
#         # Use the state_to_string method to convert the current state to a string
#         state_str = self.state_to_string(state)

#         # Implement epsilon-greedy strategy here
#         if np.random.rand() < self.epsilon:
#             # Exploration: Choose a random action
#             return random.choice(self.word_list)
#         else:
#             # Exploitation: Choose the best action based on the current Q-table
#             return self.choose_best_action(state_str)

#     def choose_best_action(self, state_str):
#         # Find the action with the highest Q-value for the current state
#         best_action = None
#         max_q_value = float('-inf')

#         for action in self.word_list:
#             q_value = self.q_table.get((state_str, action), 0)
#             if q_value > max_q_value:
#                 max_q_value = q_value
#                 best_action = action

#         return best_action if best_action else random.choice(self.word_list)

# def run_simulation(agent, environment, episodes):
#     total_rewards = []  # To store rewards from each episode for averaging

#     for episode in range(episodes):
#         current_state = environment.reset()  # Start a new game
#         total_reward = 0
#         done = False

#         while not done:
#             action = agent.guess(current_state)  # Agent makes a guess based on the current state
#             next_state, reward, done = environment.step(action)  # Environment processes the guess
#             agent.learn(current_state, action, reward, next_state)  # Agent updates its Q-table

#             total_reward += reward
#             current_state = next_state  # Update the state

#             if done:
#                 print(f"Game {episode + 1}: Word was {environment.target_word}. Last guess was '{action}' with reward {reward}.")

#         total_rewards.append(total_reward)  # Store the total reward for this episode

#     average_reward = np.mean(total_rewards)
#     print(f"Average reward after {episodes} episodes: {average_reward:.2f}")

#     return average_reward  # Optionally return the average reward

# # Assuming you have already defined and imported WordleEnvironment and QLearningWordleAgent

# # Initialize the environment and the agent
# environment = WordleEnvironment(word_list)
# agent = QLearningWordleAgent(word_list, epsilon=0.1, learning_rate=0.1, discount_factor=0.9)

# # Run the simulation for a desired number of episodes
# episodes = 1000  # You can adjust this number based on how long you want to train
# run_simulation(agent, environment, episodes)
