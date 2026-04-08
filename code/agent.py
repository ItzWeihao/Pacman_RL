import pickle
import random
from collections import defaultdict

class Agent:
    def __init__(self, name, epsilon, alpha, gamma, nu):
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.Q = defaultdict(float)
        self.total_iterations = 0

    def get_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        return max(valid_actions, key=lambda a: self.Q[(state, a)])

    def update(self, state, action, reward, next_state, valid_next_actions):
        max_q_next = max(self.Q[(next_state, a)] for a in valid_next_actions)
        self.Q[(state, action)] += self.alpha * (
            reward + self.gamma * max_q_next - self.Q[(state, action)]
        )

    def save(self, path="pacman_qtable"):
        data = {
            "Q": dict(self.Q),
            "total_iterations": self.total_iterations,
            "epsilon": self.epsilon
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path="pacman_qtable"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.Q = defaultdict(float, data["Q"])
            self.total_iterations = data["total_iterations"]
            self.epsilon = data["epsilon"]
            print(f"Resuming from iteration {self.total_iterations} with epsilon {self.epsilon}")
        except FileNotFoundError:
            print(f"No save file found, starting fresh")