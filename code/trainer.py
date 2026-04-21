from agent import Agent
from nodes import *
from run import GameController
from state import get_state
from constants import *

TRAINING = True
RENDER = True
TOTAL_REWARD_GAIN = 0

# 2300 iterations; -702,56 avg rewards
# 2400 iterations; -688.52 avg rewards

def nearest_pellet_distance(pacman, pellets):
    if not pellets.pelletList:
        return 0
    return min(
        (p.position - pacman.position).magnitudeSquared()
        for p in pellets.pelletList
    )

def nearest_ghost_distance(pacman, ghosts):
    dangerous_ghosts = [g for g in ghosts if g.mode.current is not FREIGHT]
    if not dangerous_ghosts:
        return float('inf')
    return min(
        (ghost.node.position - pacman.position).magnitudeSquared()
        for ghost in dangerous_ghosts
    )

def near_node(pacman, threshold=5):
    diff = pacman.position - pacman.target.position
    return diff.magnitudeSquared() < threshold**2

agent = Agent(
    name    = "Pacman",
    epsilon = 0.65,     # exploration rate (random percentage to take a random action)
    alpha   = 0.1,      # learning rate (how fast it learns and nudges the q-value)
    gamma   = 0.9,      # discount factor (higher = values future rewards | lower = values immediate rewards)
    nu      = 100       # number of iterations
)
agent.load()

game = GameController(training=TRAINING, render=RENDER)

game.startGame()

if TRAINING:
    for episode in range(agent.nu):
        agent.total_iterations += 1

        #if agent.total_iterations % 500 == 0:
        #    agent.epsilon = 0.65

        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        print(f"======== Iteration {agent.total_iterations} ========")
        print(f"* Agent Epsilon: {agent.epsilon}")

        state = get_state(game.pacman, game.ghosts, game.pellets)
        action = agent.get_action(state, [LEFT, RIGHT])

        time_penalty = 0
        TIME_PENALTY_INTERVAL = 1.0
        total_reward = 0

        while not game.over:
            prev_score = game.score
            prev_lives = game.lives
            prev_level = game.level
            prev_pellet_distance = nearest_pellet_distance(game.pacman, game.pellets)
            prev_ghost_distance = nearest_ghost_distance(game.pacman, game.ghosts)

            game.update()

            curr_pellet_distance = nearest_pellet_distance(game.pacman, game.pellets)
            curr_ghost_distance = nearest_ghost_distance(game.pacman, game.ghosts)

            reward = game.score - prev_score

            if game.lives < prev_lives:
                reward -= 500

            if game.level != prev_level:
                reward += 1000

            if curr_ghost_distance < prev_ghost_distance:
                reward -= 50

            if curr_pellet_distance < prev_pellet_distance:
                reward += 5

            next_state = get_state(game.pacman, game.ghosts, game.pellets)
            next_action = agent.get_action(next_state, game.pacman.validDirections())

            if game.pacman.overshotTarget():
                game.pacman.node = game.pacman.target

                if game.pacman.node.neighbors[PORTAL] is not None:
                    game.pacman.node = game.pacman.node.neighbors[PORTAL]

                game.pacman.target = game.pacman.getNewTarget(next_action)

                if game.pacman.target is not game.pacman.node:
                    game.pacman.direction = next_action
                else:
                    game.pacman.direction = STOP
                    next_action = agent.get_action(next_state, game.pacman.validDirections())
                    game.pacman.target = game.pacman.getNewTarget(next_action)

                if game.pacman.target is game.pacman.node:
                    game.pacman.direction = STOP
                    next_action = agent.get_action(next_state, game.pacman.validDirections())

                game.pacman.setPosition()

                game.pacman.direction = next_action

                agent.update(state, action, reward, next_state, game.pacman.validDirections())

                state = next_state
                action = next_action

                total_reward += reward

        print(f"* Total reward: {total_reward} \n")
        TOTAL_REWARD_GAIN += total_reward
        game.over = False
        game.restartGame()
    print(f"* Reward Average for the past {agent.nu} iterations: {TOTAL_REWARD_GAIN / agent.nu} \n")
    agent.save()
else:
    while not game.over:
        game.update()