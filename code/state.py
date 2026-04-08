from constants import *

GHOST_DANGER_RADIUS = 150

def get_state(pacman, ghosts, pellets):
    pacman_node = (int(pacman.node.position.x), int(pacman.node.position.y))
    pacman_direction = pacman.direction

    danger_up = any(
        ghost.node.position.y < pacman.position.y and
        (ghost.node.position - pacman.position).magnitudeSquared() < GHOST_DANGER_RADIUS**2
        for ghost in ghosts if ghost.mode.current is not FREIGHT
    )
    danger_down = any(
        ghost.node.position.y > pacman.position.y and
        (ghost.node.position - pacman.position).magnitudeSquared() < GHOST_DANGER_RADIUS**2
        for ghost in ghosts if ghost.mode.current is not FREIGHT
    )
    danger_left = any(
        ghost.node.position.x < pacman.position.x and
        (ghost.node.position - pacman.position).magnitudeSquared() < GHOST_DANGER_RADIUS**2
        for ghost in ghosts if ghost.mode.current is not FREIGHT
    )
    danger_right = any(
        ghost.node.position.x > pacman.position.x and
        (ghost.node.position - pacman.position).magnitudeSquared() < GHOST_DANGER_RADIUS**2
        for ghost in ghosts if ghost.mode.current is not FREIGHT
    )

    if pellets.pelletList:
        nearest = min(pellets.pelletList, key=lambda p: (p.position - pacman.position).magnitudeSquared())
        pellet_dir = (
            int(nearest.position.x > pacman.position.x),
            int(nearest.position.y > pacman.position.y)
        )
    else:
        pellet_dir = (-1, -1)

    power_pellets = [p for p in pellets.pelletList if p.name == POWERPELLET]
    if power_pellets:
        nearest_pp = min(power_pellets, key=lambda p: (p.position - pacman.position).magnitudeSquared())
        pp_nearby = (nearest_pp.position - pacman.position).magnitudeSquared() < GHOST_DANGER_RADIUS**2
    else:
        pp_nearby = False

    ghost_edible = any(ghost.mode.current is FREIGHT for ghost in ghosts)

    return (
        pacman_node,
        pacman_direction,
        danger_up,
        danger_down,
        danger_left,
        danger_right,
        pellet_dir,
        pp_nearby,
        ghost_edible,
    )