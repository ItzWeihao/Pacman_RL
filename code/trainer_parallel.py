import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import threading
from agent import Agent
from run import GameController
from state import get_state
from constants import *

TRAINING = True
NUM_INSTANCES = 4

lock = threading.Lock()

TOTAL_REWARD_GAIN = 0

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

def run_episode(agent, episode_num):
    game = GameController(training=True, render=False)
    game.startGame()
    print(f"   [Instance {episode_num}] Starting episode")

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

        reward = game.score - prev_score

        curr_pellet_distance = nearest_pellet_distance(game.pacman, game.pellets)
        curr_ghost_distance = nearest_ghost_distance(game.pacman, game.ghosts)

        power_pellet_eaten = prev_score + 50 == game.score
        ghost_nearby = nearest_ghost_distance(game.pacman, game.ghosts) < 150 ** 2
        ghost_edible = any(ghost.mode.current is FREIGHT for ghost in game.ghosts)

        if power_pellet_eaten and not ghost_nearby:
            reward -= 300

        if power_pellet_eaten and ghost_nearby:
            reward += 100

        if ghost_edible and curr_ghost_distance < prev_ghost_distance:
            reward += 100

        if game.lives < prev_lives:
            reward -= 500

        if game.level != prev_level:
            reward += 1000

        if curr_ghost_distance < prev_ghost_distance:
            reward -= 50

        if curr_pellet_distance < prev_pellet_distance:
            reward += 5

        time_penalty += 1/60
        if time_penalty >= TIME_PENALTY_INTERVAL:
            reward -= 1
            time_penalty = 0

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

            with lock:
                agent.update(state, action, reward, next_state, game.pacman.validDirections())

            state = next_state
            action = next_action
            total_reward += reward

    print(f"   [Instance {episode_num}] Finished | Reward: {total_reward}")
    return total_reward

agent = Agent(
    name    = "Pacman",
    epsilon = 0.65,
    alpha   = 0.1,
    gamma   = 0.9,
    nu      = 150
)
agent.load()

if TRAINING:
    for episode in range(0, agent.nu, NUM_INSTANCES):
        threads = []
        results = [None] * NUM_INSTANCES

        for i in range(NUM_INSTANCES):
            agent.total_iterations += 1
            agent.epsilon = max(0.01, agent.epsilon * 0.995)
            print(f"======== Iteration {agent.total_iterations} ========")
            print(f"* Agent Epsilon: {agent.epsilon:.4f}")


            def thread_fn(i, iteration):
                results[i] = run_episode(agent, iteration)


            t = threading.Thread(target=thread_fn, args=(i, agent.total_iterations))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        avg_reward = sum(results) / NUM_INSTANCES
        TOTAL_REWARD_GAIN += avg_reward
        print(f"* Avg reward across {NUM_INSTANCES} instances: {avg_reward:.2f}")
        print(f"* Running total average: {TOTAL_REWARD_GAIN / agent.total_iterations:.2f}")

    print(f"* Reward average for the past {agent.nu} iterations: {TOTAL_REWARD_GAIN / agent.nu:.2f}")
    agent.save()