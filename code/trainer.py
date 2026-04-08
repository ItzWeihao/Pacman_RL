from agent import Agent
from nodes import *
from run import GameController
from state import get_state
from constants import *

TRAINING = True
RENDER = True

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
    epsilon = 0.65,
    alpha   = 0.1,
    gamma   = 0.9,
    nu      = 100
)
agent.load()

game = GameController(training=TRAINING, render=RENDER)
game.startGame()

if TRAINING:
    for episode in range(agent.nu):
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        print(f"agent epsilon: {agent.epsilon}")
        agent.total_iterations += 1
        print(f"Iteration {agent.total_iterations}")

        state = get_state(game.pacman, game.ghosts, game.pellets)
        action = agent.get_action(state, [LEFT, RIGHT])

        time_penalty = 0
        TIME_PENALTY_INTERVAL = 1.0
        total_reward = 0

        while not game.over:
            prev_score = game.score
            prev_lives = game.lives
            prev_pellet_distance = nearest_pellet_distance(game.pacman, game.pellets)
            prev_ghost_distance = nearest_ghost_distance(game.pacman, game.ghosts)

            game.update()

            curr_pellet_distance = nearest_pellet_distance(game.pacman, game.pellets)
            curr_ghost_distance = nearest_ghost_distance(game.pacman, game.ghosts)

            reward = game.score - prev_score

            if game.lives < prev_lives:
                reward -= 500

            if curr_ghost_distance < prev_ghost_distance:
                reward -= 50

            if curr_pellet_distance < prev_pellet_distance:
                reward += 5
            else:
                reward -= 1

            #time_penalty += 1/60
            #if time_penalty >= TIME_PENALTY_INTERVAL:
            #    reward -= 1
            #    time_penalty = 0

            if game.pacman.overshotTarget():
                next_state = get_state(game.pacman, game.ghosts, game.pellets)
                next_action = agent.get_action(next_state, game.pacman.validDirections())

                print(next_action)

                agent.update(state, action, reward, next_state, game.pacman.validDirections())

                game.pacman.target = game.pacman.node.neighbors[next_action]
                game.pacman.direction = next_action

                state = next_state
                action = next_action

            total_reward += reward

        print(f"   Total reward: {total_reward} /n")
        game.over = False
        game.restartGame()
    agent.save()
else:
    while not game.over:
        game.update()