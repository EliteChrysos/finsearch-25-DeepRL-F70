import os
os.environ["KERAS_BACKEND"] = "tensorflow" 
#pip install keras tensorflow gym matplotlib
#pip install "gymnasium[classic_control]"" # For classic control environments

import keras
from keras import layers
import tensorflow as tf
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Create environment
env = gym.make("Pendulum-v1")
num_states = env.observation_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Size of State Space -> {}".format(num_states))
print("Max Value of Action -> {}".format(upper_bound))
print("Min Value of Action -> {}".format(lower_bound))

# Improved DQN Parameters
NUM_ACTIONS = 21  # More granular action space
ACTION_SPACE = np.linspace(lower_bound, upper_bound, NUM_ACTIONS)
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON = 0.9  # Start with high exploration
EPSILON_MIN = 0.05  # Keep some exploration
EPSILON_DECAY = 0.9995  # Slower decay
MEMORY_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 50  # More frequent updates

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.train_step_count = 0
        
        # Neural networks - Improved architecture
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        
        # Optimizer with gradient clipping
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        
    def build_model(self):
        """Build the neural network for DQN with improved architecture"""
        inputs = layers.Input(shape=(self.state_size,))
        
        # Normalization layer
        x = layers.BatchNormalization()(inputs)
        
        # Hidden layers with dropout
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        q_values = self.q_network(state_tensor, training=False)
        return np.argmax(q_values[0])
    
    def get_continuous_action(self, action_idx):
        """Convert discrete action index to continuous action value"""
        return ACTION_SPACE[action_idx]
    
    def reward_shaping(self, state, action, reward, next_state):
        """Improved reward shaping for better learning"""
        # Original reward is based on angle, angular velocity, and action
        # Let's add some shaping to encourage upright position
        
        cos_theta = state[0]  # cosine of angle
        sin_theta = state[1]  # sine of angle  
        angular_velocity = state[2]  # angular velocity
        
        # Reward for being upright (cos_theta close to 1)
        upright_reward = cos_theta
        
        # Penalty for high angular velocity
        velocity_penalty = -0.1 * angular_velocity**2
        
        # Small penalty for large actions
        action_penalty = -0.01 * action**2
        
        # Combine rewards
        shaped_reward = reward + upright_reward + velocity_penalty + action_penalty
        
        return shaped_reward
    
    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Single training step with Double DQN"""
        with tf.GradientTape() as tape:
            # Current Q values
            current_q_values = self.q_network(states, training=True)
            current_q_values = tf.gather(current_q_values, actions, batch_dims=1)
            
            # Double DQN: Use main network to select actions, target network to evaluate
            next_q_values_main = self.q_network(next_states, training=False)
            next_actions = tf.argmax(next_q_values_main, axis=1)
            
            next_q_values_target = self.target_network(next_states, training=False)
            next_q_values = tf.gather(next_q_values_target, next_actions, batch_dims=1)
            
            # Target Q values
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
            
            # Huber loss for stability
            loss = keras.losses.huber(target_q_values, current_q_values)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < BATCH_SIZE:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, BATCH_SIZE)
        
        # Separate batch into components
        states = np.array([experience[0] for experience in batch], dtype=np.float32)
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in batch], dtype=np.float32)
        dones = np.array([experience[4] for experience in batch], dtype=np.float32)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        dones = tf.convert_to_tensor(dones)
        
        # Train
        loss = self.train_step(states, actions, rewards, next_states, dones)
        
        # # Decay epsilon
        # if self.epsilon > EPSILON_MIN:
        #     self.epsilon *= EPSILON_DECAY
        
        self.train_step_count += 1
        
        return loss

# Initialize agent
agent = DQNAgent(num_states, NUM_ACTIONS)

# Training parameters
EPISODES = 1000  # More episodes for better convergence
UPDATE_TARGET_EVERY = TARGET_UPDATE_FREQ

# Training loop
scores = []
avg_scores = []
losses = []
best_avg_score = -np.inf

print("Starting training...")
print(f"Action space: {NUM_ACTIONS} actions from {lower_bound:.1f} to {upper_bound:.1f}")

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    episode_losses = []
    
    while not done:
        # Choose action
        action_idx = agent.act(state, training=True)
        continuous_action = agent.get_continuous_action(action_idx)
        
        # Take action in environment
        next_state, reward, done, truncated, _ = env.step([continuous_action])
        done = done or truncated
        
        # Apply reward shaping
        shaped_reward = agent.reward_shaping(state, continuous_action, reward, next_state)
        
        # Store experience
        agent.remember(state, action_idx, shaped_reward, next_state, done)
        
        # Train agent
        if len(agent.memory) > BATCH_SIZE:
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss.numpy())
        
        # Update state
        state = next_state
        total_reward += reward  # Use original reward for scoring
        step += 1
        
        # Limit episode length
        if step > 200:
            break
    
    # Update target network
    if episode % UPDATE_TARGET_EVERY == 0:
        agent.update_target_network()
    
    # Record scores
    scores.append(total_reward)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    
    # Track losses
    if episode_losses:
        losses.append(np.mean(episode_losses))
    
    # Save best model
    if avg_score > best_avg_score:
        best_avg_score = avg_score
        agent.q_network.save_weights("dqn_pendulum_best.weights.h5")
    
    # Print progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Score: {total_reward:.2f}, "
              f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # Early stopping if converged
        if len(scores) > 200 and avg_score > -150:
            print(f"Converged at episode {episode}!")
            break
    # Decay epsilon
    if agent.epsilon > EPSILON_MIN:
        agent.epsilon *= EPSILON_DECAY

print("Training completed!")
print(f"Best average score: {best_avg_score:.2f}")

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(scores, alpha=0.6)
plt.plot(avg_scores, 'r-', linewidth=2)
plt.title('Episode Scores')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.legend(['Episode Score', 'Average Score'])

plt.subplot(1, 3, 2)
plt.plot(avg_scores)
plt.title('Average Scores (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.axhline(y=-150, color='r', linestyle='--', label='Target')
plt.legend()

plt.subplot(1, 3, 3)
if losses:
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Test the trained agent
print("\nTesting trained agent...")
test_episodes = 5

# Load best model
try:
    agent.q_network.load_weights("dqn_pendulum_best.weights.h5")
    print("Loaded best model weights")
except:
    print("Using current model weights")

# Test with rendering
try:
    test_env = gym.make("Pendulum-v1", render_mode="human")
except:
    print("Display not available, running without rendering")
    test_env = gym.make("Pendulum-v1")

agent.epsilon = 0  # No exploration during testing

for episode in range(test_episodes):
    state, _ = test_env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < 200:
        # Choose best action (no exploration)
        action_idx = agent.act(state, training=False)
        continuous_action = agent.get_continuous_action(action_idx)
        
        # Take action
        next_state, reward, done, truncated, _ = test_env.step([continuous_action])
        done = done or truncated
        
        state = next_state
        total_reward += reward
        step += 1
    
    print(f"Test Episode {episode + 1}: Score = {total_reward:.2f}")

test_env.close()

# Save final model
agent.q_network.save_weights("dqn_pendulum_final.weights.h5")
print("Final model weights saved!")

# Function to load and use the trained model
def load_trained_agent():
    """Load a trained DQN agent"""
    loaded_agent = DQNAgent(num_states, NUM_ACTIONS)
    try:
        loaded_agent.q_network.load_weights("dqn_pendulum_best.weights.h5")
        print("Loaded best model")
    except:
        loaded_agent.q_network.load_weights("dqn_pendulum_final.weights.h5")
        print("Loaded final model")
    loaded_agent.epsilon = 0  # No exploration for trained agent
    return loaded_agent

print("\nTraining summary:")
print(f"Final average score: {avg_scores[-1]:.2f}")
print(f"Best average score: {best_avg_score:.2f}")
print(f"Action discretization: {NUM_ACTIONS} actions from {lower_bound} to {upper_bound}")
print(f"Target score for good performance: > -150")

# Additional analysis
if best_avg_score > -200:
    print("✓ Agent learned to control the pendulum reasonably well!")
else:
    print("⚠ Agent may need more training or hyperparameter tuning.")