# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


import util

def manhattanDistance(xy1, xy2):
    """
    Returns the Manhattan distance between two points.
    """
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class BetterFeatureExtractor(SimpleExtractor):
    """
    Extends the SimpleExtractor by adding features related to capsules, score,
    ghost positions, Pacman's position and direction, and food pellets.
    """

    def __init__(self):
        self.max_feature_values = {
            "bias": 1.0,
            '#-of-ghosts-1-step-away' : -1.0,
            'eats-food':1.0,
            "closest-capsule": 0.005,
            "closest-ghost": -0.008,
            "distance-to-ghost0": -0.004,
            "distance-to-ghost1": -0.004,
            "distance-to-ghost2": -0.004,
            "distance-to-ghost3": -0.004,
            "closest-food": 0.07,
            "score": 0.03,# Maximum value for the direction feature
        }

    def getFeatures(self, state, action):
        features = SimpleExtractor.getFeatures(self, state, action)

        # Get the positions of capsules
        capsules = state.getCapsules()

        # Compute the distance to the nearest capsule
        capsule_distances = [manhattanDistance(pos, state.getPacmanPosition()) for pos in capsules]
        if capsule_distances:
            features["closest-capsule"] = min(capsule_distances)
        else:
            features["closest-capsule"] = 0.0

        # Get the current score
        features["score"] = state.getScore()+0.01
        self.max_feature_values["score"] = state.getScore()  # Update the maximum score seen

        # Get Pacman's state
        pacman_state = state.getPacmanState()
        pacman_position = pacman_state.getPosition()

        # Compute the distance to the nearest ghost
        ghost_positions = state.getGhostPositions()
        ghost_distances = [manhattanDistance(pos, pacman_position) for pos in ghost_positions]
        if ghost_distances:
            features["closest-ghost"] = min(ghost_distances)
        else:
            features["closest-ghost"] = 0.0

        ghost_states = state.getGhostStates()
        for i, ghost_state in enumerate(ghost_states):
            ghost_position = ghost_state.getPosition()
            distance_to_ghost = manhattanDistance(pacman_position, ghost_position)
            features["distance-to-ghost{}".format(i)] = distance_to_ghost

        # Compute the distance to the nearest food pellet
        food = state.getFood()
        food_positions = [(x, y) for x in range(food.width) for y in range(food.height) if food[x][y]]
        food_distances = [manhattanDistance(pos, pacman_position) for pos in food_positions]
        if food_distances:
            features["closest-food"] = min(food_distances)
        else:
            features["closest-food"] = 0.0

        # Normalize features
        for feature in features:
            if self.max_feature_values[feature] is not None and (feature != 'score'):
                features[feature] /= self.max_feature_values[feature]

        return features
