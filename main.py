import numpy as np
import random
import time
from typing import Tuple, List
import tkinter as tk
from datetime import datetime

##########
# Maze Generation and Visualization
##########
class MazeGenerator:
    @staticmethod 
    def generate_maze(size: int = 6, complexity: float = 0.3) -> np.ndarray:
        maze = np.zeros((size, size))
        num_walls = int((size * size) * complexity)
        wall_positions = random.sample([(i, j) for i in range(size) for j in range(size)], num_walls)
        
        for pos in wall_positions:
            maze[pos] = 1
            
        maze[0, 0] = 0  # Start
        maze[size-1, size-1] = 2  # Goal
        
        # initialise every time
        current = (0, 0)
        goal = (size-1, size-1)
        path = MazeGenerator._find_path_to_goal(current, goal, maze)
        
        for pos in path:
            maze[pos] = 0
        maze[size-1, size-1] = 2
        
        return maze
    
    @staticmethod
    def _find_path_to_goal(start: Tuple[int, int], goal: Tuple[int, int], 
                          maze: np.ndarray) -> List[Tuple[int, int]]:
        path = [start]
        current = start
        while current != goal:
            row, col = current
            if row < goal[0]:
                current = (row + 1, col)
            elif col < goal[1]:
                current = (row, col + 1)
            path.append(current)
        return path

class MazeVisualizer:
    def __init__(self, size: int):
        self.size = size
        self.cell_size = 60
        self.window = tk.Tk()
        self.window.title("Q-Learning Maze Solver")

        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(padx=10, pady=10)

        canvas_size = self.cell_size * size
        self.canvas = tk.Canvas(self.main_frame, width=canvas_size, height=canvas_size)
        self.canvas.pack(side=tk.LEFT, padx=10)

        self.log_frame = tk.Frame(self.main_frame)
        self.log_frame.pack(side=tk.LEFT, padx=10)
        
        self.log_text = tk.Text(self.log_frame, width=50, height=20)
        self.log_text.pack()

        self.status_frame = tk.Frame(self.window)
        self.status_frame.pack(pady=5)
        
        self.episode_label = tk.Label(self.status_frame, text="Episode: 0")
        self.episode_label.pack(side=tk.LEFT, padx=10)
        
        self.steps_label = tk.Label(self.status_frame, text="Steps: 0")
        self.steps_label.pack(side=tk.LEFT, padx=10)
        
        self.epsilon_label = tk.Label(self.status_frame, text="Epsilon: 1.00")
        self.epsilon_label.pack(side=tk.LEFT, padx=10)
         
        self.points_label = tk.Label(self.status_frame, text="Points: 0")
        self.points_label.pack(side=tk.LEFT, padx=10)
        
    def update_maze(self, maze: np.ndarray, current_pos: Tuple[int, int]):
        self.canvas.delete("all")
        
        colors = {
            0: "white",  # empty
            1: "gray",   # wall
            2: "green",  # goal
            3: "red"     # agent
        }
        
        for i in range(self.size):
            for j in range(self.size):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                cell_value = maze[i, j]
                color = colors[cell_value]
                
                if (i, j) == current_pos:
                    color = colors[3]
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
        
        self.window.update()
    
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.window.update()
    
    def update_status(self, episode: int, steps: int, epsilon: float, points: float):
        self.episode_label.config(text=f"Episode: {episode}")
        self.steps_label.config(text=f"Steps: {steps}")
        self.epsilon_label.config(text=f"Epsilon: {epsilon:.2f}")
        self.points_label.config(text=f"Points: {points:.2f}")
        self.window.update()

############
# Q-Learning Agent and Environment
############
class MazeEnvironment:
    def __init__(self, size: int = 6):
        self.size = size
        self.maze = None
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.goal_pos = (size-1, size-1)
        self.visualizer = MazeVisualizer(size)
        self.reset()
        
    def reset(self) -> Tuple[int, int]:
        self.maze = MazeGenerator.generate_maze(self.size)
        self.current_pos = self.start_pos
        self.visualizer.update_maze(self.maze, self.current_pos)
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        new_pos = (
            self.current_pos[0] + moves[action][0],
            self.current_pos[1] + moves[action][1]
        )
        
        if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size and 
            self.maze[new_pos] != 1):
            self.current_pos = new_pos
            
        self.visualizer.update_maze(self.maze, self.current_pos)
        
        if self.current_pos == self.goal_pos:
            return self.current_pos, 100, True
        elif new_pos != self.current_pos:
            return self.current_pos, -1, False
        else:
            return self.current_pos, -5, False
        
##########
# Q-Learning Agent
##########

class QLearningAgent: # dont change annything here 
    def __init__(self, state_size: int, action_size: int):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.1
        self.gamma = 0.95
        
    def get_action(self, state: Tuple[int, int]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return np.argmax(self.q_table[state])
    # This is where Q-table is updated
    # The Q-table is a 3D array where the first two dimensions are the state (x, y) and the third dimension is the action
    # Example Q-table update:
# Assume:
# - state = (1, 2)
# - action = 0
# - reward = 10
# - next_state = (1, 3)
# - learning_rate = 0.1
# - gamma = 0.9
# - old_value (Q[state][action]) = 5
# - max Q-value for next_state = 15

# Step 1: Get current Q-value
# old_value = q_table[(1,2)][0] = 5

# Step 2: Find max Q-value for next state
# next_max = max(q_table[(1,3)]) = 15

# Step 3: Calculate new Q-value
# new_value = (1-0.1)*5 + 0.1*(10 + 0.9*15)
#          = 0.9*5 + 0.1*(10 + 13.5)
#          = 4.5 + 0.1*23.5
#          = 4.5 + 2.35
#          = 6.85

# Step 4: Update Q-table
# q_table[(1,2)][0] = 6.85

# Step 5: Decay epsilon (if applicable)
# epsilon = epsilon * epsilon_decay
    def update(self, state: Tuple[int, int], action: int, 
               reward: float, next_state: Tuple[int, int]):
        old_value = self.q_table[state][action]     # Q(s,a)
        
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train(episodes: int = 200, maze_size: int = 6):
    env = MazeEnvironment(maze_size)
    agent = QLearningAgent(maze_size, 4)
    
    success_rate = []
    window_size = 20
    successes = 0
    
    # Track cumulative rewards across episodes (not working rn)
    cumulative_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        env.visualizer.log_message(f"Episode {episode + 1} started")
        
        while not done and steps < 200:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            # Internal if statement checking if the next state is not a wall
            # and if the next state is not the same as the current state
            
            if next_state != state:
                agent.update(state, action, reward, next_state)
                state = next_state
                steps += 1
            
            total_reward += reward
            env.visualizer.update_status(episode + 1, steps, agent.epsilon, total_reward)
            time.sleep(0.1)  # Slow down visualization, remove to show reduction in sigma
            
        cumulative_rewards.append(total_reward)
        if done and reward > 0:
            successes += 1
            env.visualizer.log_message(
                f"Episode {episode + 1} completed successfully in {steps} steps! Total Reward: {total_reward}"
            )
        else:
            env.visualizer.log_message(
                f"Episode {episode + 1} failed after {steps} steps. Total Reward: {total_reward}"
            )
        
        if episode % window_size == 0 and episode > 0:
            success_rate.append(successes / window_size)
            env.visualizer.log_message(
                f"Success rate over last {window_size} episodes: {success_rate[-1]:.2%}. "
                f"Average Reward: {np.mean(cumulative_rewards[-window_size:]):.2f}"
            )
            successes = 0
        
        time.sleep(0.5)  # Pause between episodes
    
    env.visualizer.window.mainloop()

if __name__ == "__main__":
    train(episodes=200, maze_size=6)