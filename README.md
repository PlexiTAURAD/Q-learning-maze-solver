# Q-Learning Maze Solver with Visualization

This project demonstrates a simple **Q-Learning** agent that learns to navigate a randomly generated maze. It includes a **Tkinter-based visualizer** that displays the maze, the agent's movements, and the learning process over several episodes. The goal is for the agent to reach the goal (green cell) from the start (top-left corner) while avoiding walls (gray cells).

## Features

- **Maze Generation**: Randomly generates a maze with customizable size and complexity.
- **Q-Learning**: Implements the Q-learning algorithm for solving the maze.
- **Visualization**: Uses **Tkinter** to visualize the agent’s movements and learning process in real-time.
- **Logging**: Logs each episode, steps taken, and agent performance over time.
- **Dynamic Maze Updates**: Updates the maze layout and the agent's position after every step.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/PlexiTAURAD/q-learning-maze-solver.git
   cd q-learning-maze-solver
   ```

2. Install dependencies:

   Ensure you have Python installed along with the necessary libraries:

   ```bash
   pip install numpy tkinter
   ```

   For **Tkinter**, it usually comes pre-installed with Python, but if not, you may need to install it separately based on your OS.

## How to Run

Run the `train` function, which trains the Q-learning agent over several episodes:

```bash
python main.py
```

You can customize the number of episodes and maze size within the `train()` function:

```python
train(episodes=200, maze_size=6)
```

## Q-Learning Algorithm

The agent learns the best path to reach the goal by exploring the maze and updating a **Q-table** that represents state-action pairs. The Q-table is updated based on the agent’s actions, the rewards it receives, and future potential rewards.

### Parameters

- **Learning Rate (α)**: Controls how much new information overrides old information.
- **Discount Factor (γ)**: Determines how much the agent values future rewards over immediate rewards.
- **Epsilon (ε)**: Governs the balance between exploration (random actions) and exploitation (choosing the best-known action).

## Maze Environment

The maze is a 2D grid where:

- `0`: Empty cell (white)
- `1`: Wall (gray)
- `2`: Goal (green)
- `3`: Agent’s current position (red)

The agent moves in one of four possible directions: up, down, left, or right. If the agent hits a wall, it remains in the same position. The goal is to navigate from the top-left corner to the bottom-right corner.

## Visualization

The maze and the agent's movements are visualized in real-time using **Tkinter**. The agent's current position is highlighted in red, while the goal is green. The interface also displays the current episode, the number of steps taken, and the agent's epsilon value.

## Example Output

- **Console Output**: Logs the start and end of each episode and whether the agent succeeded in reaching the goal.
- **Tkinter GUI**: Displays the maze and agent’s movements in real-time.



![image](https://github.com/user-attachments/assets/6d6e276a-9c5d-4fbc-bfc5-662881701e61)


![image](https://github.com/user-attachments/assets/30c62ee5-3ef3-4b87-a85a-95d98c8ad0b3)



## Customization

You can customize several aspects of the project:

- **Maze Size**: Modify the maze size (default: 6x6) in the `train()` function.
- **Q-Learning Parameters**: Adjust the learning rate, discount factor, and epsilon values in the `QLearningAgent` class.
