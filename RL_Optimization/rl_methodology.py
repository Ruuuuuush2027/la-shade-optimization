"""
Q-Learning Agent for LA 2028 Olympics Shade Placement Optimization

This module implements the reinforcement learning approach for strategic
shade structure placement across Los Angeles.

The agent learns to place 50 shade structures across 2,650 grid points
by maximizing cumulative reward through trial-and-error learning.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import random
import matplotlib.pyplot as plt


class ShadeQLearningAgent:
    """
    Q-Learning agent that learns optimal shade placement policy.
    
    The agent operates in episodes where it sequentially places up to 50
    shade structures, learning from the rewards obtained to improve its
    policy over time.
    
    Key Concepts:
    - State: Set of already-placed shade locations (frozenset of indices)
    - Action: Selecting a grid point for next shade placement (0-2649)
    - Q(s,a): Learned value estimating long-term reward from action a in state s
    """
    
    def __init__(self, data_df, reward_function, n_shades_budget=50, 
                 alpha=0.1, gamma=0.95, epsilon=0.3):
        """
        Initialize the Q-Learning agent.
        
        Args:
            data_df (pd.DataFrame): Dataset with 2650 grid points and 84 features
            reward_function (ShadeRewardFunction): Reward function instance
            n_shades_budget (int): Maximum number of shades to place (default: 50)
            alpha (float): Learning rate (0-1, default: 0.1)
                          Controls how much new information overrides old
            gamma (float): Discount factor (0-1, default: 0.95)
                          Controls importance of future rewards
            epsilon (float): Initial exploration rate (0-1, default: 0.3)
                            Probability of random action vs. greedy action
        """
        # Dataset and reward function
        self.data = data_df
        self.reward_fn = reward_function
        self.n_points = len(data_df)
        
        # Budget constraint
        self.budget = n_shades_budget
        
        # Q-Learning hyperparameters
        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # Exploration rate (decays over time)
        self.epsilon_min = 0.01         # Minimum exploration rate
        self.epsilon_decay = 0.995      # Decay rate per episode
        
        # Q-table: maps (state, action) → value
        # State is frozenset of placed shade indices (hashable)
        # Action is grid point index (0-2649)
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Training history for analysis
        self.training_history = {
            'episode_rewards': [],      # Total reward per episode
            'episode_lengths': [],      # Number of shades placed per episode
            'epsilon_values': []        # Epsilon value per episode
        }
        
        print(f"Q-Learning Agent initialized:")
        print(f"  Grid points: {self.n_points}")
        print(f"  Shade budget: {self.budget}")
        print(f"  α (learning rate): {self.alpha}")
        print(f"  γ (discount factor): {self.gamma}")
        print(f"  ε₀ (initial exploration): {self.epsilon}")
    
    # ========================================================================
    # STATE AND ACTION MANAGEMENT
    # ========================================================================
    
    def _state_to_key(self, placed_locations):
        """
        Convert list of placed locations to hashable state representation.
        
        Uses frozenset for efficient hashing and lookup in Q-table.
        
        Args:
            placed_locations (list): List of grid point indices with shades
        
        Returns:
            frozenset: Hashable state key
        """
        return frozenset(placed_locations)
    
    def _get_available_actions(self, state):
        """
        Get list of valid actions (grid points without shades).
        
        Args:
            state (frozenset): Current state
        
        Returns:
            list: Available grid point indices
        """
        placed_set = set(state)
        return [i for i in range(self.n_points) if i not in placed_set]
    
    # ========================================================================
    # ACTION SELECTION (EPSILON-GREEDY)
    # ========================================================================
    
    def choose_action(self, state, training=True):
        """
        Select action using epsilon-greedy strategy.
        
        With probability ε: explore (random action)
        With probability (1-ε): exploit (best known action)
        
        This balances exploration of new strategies vs. exploitation of
        known good strategies.
        
        Args:
            state (frozenset): Current state
            training (bool): If True, use ε-greedy; if False, pure greedy
        
        Returns:
            int: Selected action (grid point index), or None if terminal state
        """
        # Get available actions
        available = self._get_available_actions(state)
        
        # Terminal state check: budget exhausted or no actions left
        if not available or len(state) >= self.budget:
            return None
        
        # EXPLORATION: Random action
        if training and random.random() < self.epsilon:
            return random.choice(available)
        
        # EXPLOITATION: Best known action
        # Get Q-values for all available actions
        q_values = {
            action: self.Q[state][action] 
            for action in available
        }
        
        # If no Q-values exist yet, initialize to 0 (optimistic initialization)
        if not q_values or all(v == 0 for v in q_values.values()):
            return random.choice(available)
        
        # Return action with highest Q-value
        best_action = max(q_values, key=q_values.get)
        return best_action
    
    # ========================================================================
    # Q-VALUE UPDATE (CORE LEARNING ALGORITHM)
    # ========================================================================
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                          └─────────────────────┘
                               TD Error
        
        This is the CORE of Q-learning. The agent learns to estimate the
        long-term value of taking action a in state s.
        
        Args:
            state (frozenset): Current state
            action (int): Action taken
            reward (float): Immediate reward received R(s, a)
            next_state (frozenset): Resulting state after action
        """
        # Current Q-value (defaults to 0 if never seen)
        current_q = self.Q[state][action]
        
        # Estimate of future value: max Q(s', a') over all next actions
        if len(next_state) < self.budget:
            # Non-terminal state: look ahead to best future action
            next_actions = self._get_available_actions(next_state)
            if next_actions:
                max_next_q = max(
                    self.Q[next_state][a] for a in next_actions
                )
            else:
                max_next_q = 0.0
        else:
            # Terminal state: no future rewards
            max_next_q = 0.0
        
        # Temporal Difference (TD) target: r + γ·max Q(s',a')
        td_target = reward + self.gamma * max_next_q
        
        # TD error: how much we were wrong
        td_error = td_target - current_q
        
        # Update Q-value: move toward TD target
        self.Q[state][action] = current_q + self.alpha * td_error
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    def train(self, n_episodes=1000, verbose=True):
        """
        Train the agent using Q-learning over multiple episodes.
        
        Each episode:
        1. Start with empty state (no shades)
        2. Sequentially place up to 50 shades
        3. For each placement:
           - Choose action (ε-greedy)
           - Get reward from reward function
           - Update Q-value
           - Move to next state
        4. Decay epsilon (reduce exploration)
        
        Args:
            n_episodes (int): Number of training episodes (default: 1000)
            verbose (bool): Print progress every 100 episodes
        
        Returns:
            dict: Training history with rewards, lengths, epsilon values
        """
        print(f"\n{'='*70}")
        print(f"TRAINING Q-LEARNING AGENT")
        print(f"{'='*70}")
        print(f"Episodes: {n_episodes}")
        print(f"Budget per episode: {self.budget} shades")
        
        for episode in range(n_episodes):
            # ================================================================
            # EPISODE INITIALIZATION
            # ================================================================
            state = frozenset()  # Empty state: no shades placed yet
            episode_reward = 0.0
            episode_length = 0
            
            # ================================================================
            # EPISODE LOOP: Place shades sequentially
            # ================================================================
            for step in range(self.budget):
                # Step 1: Choose action (which grid point to place shade)
                action = self.choose_action(state, training=True)
                
                # Check for terminal state
                if action is None:
                    break
                
                # Step 2: Calculate immediate reward R(s, a)
                # This calls the reward function with current state and action
                reward = self.reward_fn.calculate_reward(
                    list(state),  # Convert frozenset to list for reward function
                    action
                )
                
                # Step 3: Transition to next state
                # Add the new shade location to state
                next_state = frozenset(list(state) + [action])
                
                # Step 4: Update Q-value (LEARNING HAPPENS HERE)
                self.update_q_value(state, action, reward, next_state)
                
                # Step 5: Move to next state
                state = next_state
                
                # Track episode statistics
                episode_reward += reward
                episode_length += 1
            
            # ================================================================
            # EPISODE CLEANUP
            # ================================================================
            
            # Store episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['epsilon_values'].append(self.epsilon)
            
            # Decay epsilon: reduce exploration over time
            # Early episodes: explore more (high ε)
            # Later episodes: exploit more (low ε)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # ================================================================
            # PROGRESS REPORTING
            # ================================================================
            if verbose and (episode + 1) % 100 == 0:
                # Calculate moving average over last 100 episodes
                recent_rewards = self.training_history['episode_rewards'][-100:]
                avg_reward = np.mean(recent_rewards)
                
                recent_lengths = self.training_history['episode_lengths'][-100:]
                avg_length = np.mean(recent_lengths)
                
                print(f"Episode {episode + 1:4d}/{n_episodes} | "
                      f"Avg Reward: {avg_reward:6.3f} | "
                      f"Avg Length: {avg_length:4.1f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Q-table size: {len(self.Q)}")
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total episodes: {n_episodes}")
        print(f"States explored: {len(self.Q)}")
        print(f"Final ε: {self.epsilon:.4f}")
        
        return self.training_history
    
    # ========================================================================
    # POLICY EXTRACTION
    # ========================================================================
    
    def get_optimal_policy(self, max_shades=None):
        """
        Extract optimal shade placement policy from learned Q-values.
        
        Uses pure greedy selection (no exploration) to get the best policy
        the agent has learned.
        
        This is what we use after training to get actual recommendations.
        
        Args:
            max_shades (int): Maximum shades to place (default: budget)
        
        Returns:
            list: Ordered list of grid point indices for optimal shade placement
        """
        max_shades = max_shades or self.budget
        
        state = frozenset()
        policy = []
        
        print(f"\nExtracting optimal policy (placing {max_shades} shades)...")
        
        for step in range(max_shades):
            # Use pure greedy selection (training=False)
            action = self.choose_action(state, training=False)
            
            if action is None:
                print(f"  Policy terminated early at {len(policy)} shades")
                break
            
            # Add to policy
            policy.append(action)
            
            # Update state
            state = frozenset(list(state) + [action])
        
        print(f"  ✓ Optimal policy extracted: {len(policy)} shades")
        return policy
    
    # ========================================================================
    # POLICY EVALUATION
    # ========================================================================
    
    def evaluate_policy(self, policy):
        """
        Evaluate a placement policy by calculating total cumulative reward.
        
        Simulates placing shades according to the policy and sums rewards.
        This is used to compare RL policy against baselines.
        
        Args:
            policy (list): List of grid point indices in placement order
        
        Returns:
            float: Total cumulative reward
        """
        total_reward = 0.0
        state = []
        
        for action in policy:
            # Calculate reward for this placement
            reward = self.reward_fn.calculate_reward(state, action)
            total_reward += reward
            
            # Update state
            state.append(action)
        
        return total_reward
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_training_curves(self, save_path=None):
        """
        Plot training progress curves.
        
        Creates 3 subplots:
        1. Episode rewards over time (with moving average)
        2. Episode lengths (number of shades placed)
        3. Epsilon decay (exploration rate)
        
        Args:
            save_path (str, optional): Path to save figure
        
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # ----------------------------------------------------------------
        # Subplot 1: Episode Rewards
        # ----------------------------------------------------------------
        rewards = self.training_history['episode_rewards']
        axes[0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        
        # Add moving average for smoother trend
        window = 50
        if len(rewards) >= window:
            smoothed = np.convolve(
                rewards,
                np.ones(window) / window,
                mode='valid'
            )
            axes[0].plot(range(window-1, len(rewards)), smoothed, 
                        linewidth=2, color='darkblue', 
                        label=f'{window}-episode MA')
            axes[0].legend()
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Reward Progress')
        axes[0].grid(True, alpha=0.3)
        
        # ----------------------------------------------------------------
        # Subplot 2: Episode Lengths
        # ----------------------------------------------------------------
        lengths = self.training_history['episode_lengths']
        axes[1].plot(lengths, color='green')
        axes[1].axhline(y=self.budget, color='red', linestyle='--', 
                       label=f'Budget ({self.budget})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Shades Placed')
        axes[1].set_title('Episode Length')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # ----------------------------------------------------------------
        # Subplot 3: Epsilon Decay
        # ----------------------------------------------------------------
        epsilons = self.training_history['epsilon_values']
        axes[2].plot(epsilons, color='orange')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Epsilon (ε)')
        axes[2].set_title('Exploration Rate Decay')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        return fig


# ============================================================================
# BASELINE ALGORITHMS FOR COMPARISON
# ============================================================================

class RandomBaseline:
    """
    Baseline 1: Random shade placement.
    
    Simply selects random grid points without any strategy.
    Establishes lower bound performance.
    """
    
    def __init__(self, data_df, n_shades=50):
        """
        Initialize random baseline.
        
        Args:
            data_df (pd.DataFrame): Dataset
            n_shades (int): Number of shades to place
        """
        self.data = data_df
        self.n_shades = n_shades
        self.name = "Random Baseline"
    
    def get_policy(self):
        """
        Generate random placement policy.
        
        Returns:
            list: Random selection of grid point indices
        """
        print(f"\n{self.name}: Selecting {self.n_shades} random locations...")
        policy = random.sample(range(len(self.data)), self.n_shades)
        print(f"  ✓ Policy generated")
        return policy


class GreedyByFeatureBaseline:
    """
    Baseline 2: Greedy by single feature.
    
    Selects top-N locations by a single feature value.
    Tests whether simple rules suffice.
    """
    
    def __init__(self, data_df, feature_name, n_shades=50):
        """
        Initialize greedy-by-feature baseline.
        
        Args:
            data_df (pd.DataFrame): Dataset
            feature_name (str): Feature to optimize (e.g., 'urban_heat_idx_percentile')
            n_shades (int): Number of shades to place
        """
        self.data = data_df
        self.feature_name = feature_name
        self.n_shades = n_shades
        self.name = f"Greedy-by-{feature_name}"
    
    def get_policy(self):
        """
        Generate policy by selecting top-N by feature value.
        
        Returns:
            list: Grid point indices sorted by feature value (descending)
        """
        print(f"\n{self.name}: Selecting top {self.n_shades} by {self.feature_name}...")
        
        # Sort by feature value (descending) and take top N
        sorted_indices = self.data[self.feature_name].argsort()[::-1]
        policy = sorted_indices[:self.n_shades].tolist()
        
        print(f"  ✓ Policy generated")
        return policy


class GreedyOptimizationBaseline:
    """
    Baseline 3: Greedy optimization using reward function.
    
    Iteratively selects location with highest immediate reward.
    Tests myopic placement vs. strategic planning (RL).
    
    This is the KEY baseline to beat - it uses the same reward function
    as RL but makes decisions myopically (one step at a time).
    """
    
    def __init__(self, data_df, reward_function, n_shades=50):
        """
        Initialize greedy optimization baseline.
        
        Args:
            data_df (pd.DataFrame): Dataset
            reward_function (ShadeRewardFunction): Same reward function as RL
            n_shades (int): Number of shades to place
        """
        self.data = data_df
        self.reward_fn = reward_function
        self.n_shades = n_shades
        self.name = "Greedy Optimization"
    
    def get_policy(self):
        """
        Generate policy by iteratively maximizing immediate reward.
        
        At each step:
        1. Try all available locations
        2. Calculate R(current_state, candidate_action)
        3. Select location with highest immediate reward
        4. Repeat
        
        Returns:
            list: Greedily constructed placement policy
        """
        print(f"\n{self.name}: Iteratively maximizing immediate reward...")
        print(f"  (This may take a few minutes for {self.n_shades} shades)")
        
        policy = []
        
        for i in range(self.n_shades):
            best_action = None
            best_reward = -float('inf')
            
            # Try all available actions
            available = [j for j in range(len(self.data)) if j not in policy]
            
            for action in available:
                # Calculate immediate reward R(state, action)
                reward = self.reward_fn.calculate_reward(policy, action)
                
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
            
            # Add best action to policy
            if best_action is not None:
                policy.append(best_action)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Placed {i + 1}/{self.n_shades} shades...")
        
        print(f"  ✓ Policy generated")
        return policy


# ============================================================================
# MAIN EXECUTION AND COMPARISON
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LA 2028 OLYMPICS SHADE PLACEMENT - RL OPTIMIZATION")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n[STEP 1] Loading dataset...")
    
    # PSEUDO: Load actual CSV
    # data = pd.read_csv('la_coverage_points_features.csv')
    # print(f"  ✓ Loaded {len(data)} grid points with {len(data.columns)} features")
    
    print("  [PSEUDO] Dataset: la_coverage_points_features.csv")
    print("  [PSEUDO] Shape: (2650, 84)")
    
    # ========================================================================
    # STEP 2: INITIALIZE REWARD FUNCTION
    # ========================================================================
    print("\n[STEP 2] Initializing reward function...")
    
    # PSEUDO: Create reward function
    # from reward_function import ShadeRewardFunction
    # reward_fn = ShadeRewardFunction(data)
    
    print("  [PSEUDO] Reward function initialized")
    print("  [PSEUDO] Components: heat(30%), pop(25%), access(20%), equity(15%), coverage(10%)")
    
    # ========================================================================
    # STEP 3: INITIALIZE AND TRAIN RL AGENT
    # ========================================================================
    print("\n[STEP 3] Initializing Q-Learning agent...")
    
    # PSEUDO: Create and train agent
    # agent = ShadeQLearningAgent(
    #     data, 
    #     reward_fn,
    #     n_shades_budget=50,
    #     alpha=0.1,
    #     gamma=0.95,
    #     epsilon=0.3
    # )
    
    print("  [PSEUDO] Agent parameters:")
    print("    α (learning rate) = 0.1")
    print("    γ (discount factor) = 0.95")
    print("    ε₀ (exploration) = 0.3")
    print("    Budget = 50 shades")
    
    print("\n[STEP 3a] Training agent...")
    # history = agent.train(n_episodes=1000, verbose=True)
    
    print("  [PSEUDO] Training over 1000 episodes...")
    print("  [PSEUDO] Episode  100: Avg Reward: 28.432 | ε: 0.180")
    print("  [PSEUDO] Episode  500: Avg Reward: 31.856 | ε: 0.026")
    print("  [PSEUDO] Episode 1000: Avg Reward: 33.124 | ε: 0.010")
    print("  [PSEUDO] ✓ Training complete")
    
    # ========================================================================
    # STEP 4: EXTRACT OPTIMAL POLICY
    # ========================================================================
    print("\n[STEP 4] Extracting optimal RL policy...")
    
    # optimal_policy = agent.get_optimal_policy()
    # rl_reward = agent.evaluate_policy(optimal_policy)
    
    print("  [PSEUDO] ✓ Optimal policy extracted: 50 shades")
    rl_reward = 33.124
    print(f"  [PSEUDO] RL Total Reward: {rl_reward:.3f}")
    
    # ========================================================================
    # STEP 5: COMPARE WITH BASELINES
    # ========================================================================
    print("\n[STEP 5] Comparing with baseline algorithms...")
    print("="*70)
    
    # Baseline 1: Random
    print("\n[Baseline 1] Random Placement")
    # random_baseline = RandomBaseline(data, n_shades=50)
    # random_policy = random_baseline.get_policy()
    # random_reward = agent.evaluate_policy(random_policy)
    random_reward = 22.145
    print(f"  [PSEUDO] Random Reward: {random_reward:.3f}")
    
    # Baseline 2: Greedy by UHI
    print("\n[Baseline 2] Greedy-by-UHI")
    # greedy_uhi = GreedyByFeatureBaseline(data, 'urban_heat_idx_percentile', n_shades=50)
    # greedy_uhi_policy = greedy_uhi.get_policy()
    # greedy_uhi_reward = agent.evaluate_policy(greedy_uhi_policy)
    greedy_uhi_reward = 26.783
    print(f"  [PSEUDO] Greedy-by-UHI Reward: {greedy_uhi_reward:.3f}")
    
    # Baseline 3: Greedy by Poverty
    print("\n[Baseline 3] Greedy-by-Poverty")
    # greedy_pov = GreedyByFeatureBaseline(data, 'cva_poverty', n_shades=50)
    # greedy_pov_policy = greedy_pov.get_policy()
    # greedy_pov_reward = agent.evaluate_policy(greedy_pov_policy)
    greedy_pov_reward = 24.891
    print(f"  [PSEUDO] Greedy-by-Poverty Reward: {greedy_pov_reward:.3f}")
    
    # Baseline 4: Greedy Optimization (MAIN COMPARISON)
    print("\n[Baseline 4] Greedy Optimization (using reward function)")
    # greedy_opt = GreedyOptimizationBaseline(data, reward_fn, n_shades=50)
    # greedy_opt_policy = greedy_opt.get_policy()
    # greedy_opt_reward = agent.evaluate_policy(greedy_opt_policy)
    greedy_opt_reward = 30.567
    print(f"  [PSEUDO] Greedy Optimization Reward: {greedy_opt_reward:.3f}")
    
    # ========================================================================
    # STEP 6: SUMMARY AND ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    results = {
        'Q-Learning (RL)': rl_reward,
        'Greedy Optimization': greedy_opt_reward,
        'Greedy-by-UHI': greedy_uhi_reward,
        'Greedy-by-Poverty': greedy_pov_reward,
        'Random': random_reward
    }
    
    # Sort by reward (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRanking (by total cumulative reward):")
    for rank, (method, reward) in enumerate(sorted_results, 1):
        marker = "✓ BEST" if rank == 1 else ""
        print(f"  {rank}. {method:25s}: {reward:6.3f}  {marker}")
    
    # Calculate improvements
    print("\nPerformance Improvements:")
    improvement_over_greedy = ((rl_reward - greedy_opt_reward) / greedy_opt_reward) * 100
    improvement_over_random = ((rl_reward - random_reward) / random_reward) * 100
    
    print(f"  RL vs. Greedy Optimization: +{improvement_over_greedy:.1f}%")
    print(f"  RL vs. Random:              +{improvement_over_random:.1f}%")
    
    print("\nKey Insight:")
    if improvement_over_greedy > 5:
        print("  ✓ RL significantly outperforms greedy optimization!")
        print("  ✓ Strategic planning (lookahead) provides substantial value")
        print("  ✓ Sequential placement dependencies successfully learned")
    else:
        print("  → RL shows modest improvement over greedy")
        print("  → May need hyperparameter tuning or more training episodes")
    
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    # PSEUDO: Generate visualizations
    # fig = agent.plot_training_curves(save_path='training_curves.png')
    # plt.show()
    
    print("\n[PSEUDO] Training curves plotted:")
    print("  - Episode rewards with 50-episode moving average")
    print("  - Episode lengths (shades placed per episode)")
    print("  - Epsilon decay (exploration rate)")
    print("\n[PSEUDO] Spatial visualizations would include:")
    print("  - Priority heatmap (frequency of selection during training)")
    print("  - Policy comparison map (RL vs. best baseline)")
    print("  - Coverage analysis by census tract")
    print("  - Feature importance rankings")
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)