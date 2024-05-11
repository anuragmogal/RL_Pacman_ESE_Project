# qlearningAgents.py
# ------------------
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

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
import random, util, math
from collections import defaultdict

class QLearningAgent(ReinforcementAgent):
   """
     Q-Learning Agent

     Functions you should fill in:
       - computeValueFromQValues
       - computeActionFromQValues
       - getQValue
       - getAction
       - update

     Instance variables you have access to
       - self.epsilon (exploration prob)
       - self.alpha (learning rate)
       - self.discount (discount rate)

     Functions you should use
       - self.getLegalActions(state)
         which returns legal actions for a state
   """

   def __init__(self, **args):
       "You can initialize Q-values here..."
       ReinforcementAgent.__init__(self, **args)
       self.qvalues = defaultdict(float)  # Use a defaultdict for constant-time access

   def getQValue(self, state, action):
       """
         Returns Q(state,action)
         Should return 0.0 if we have never seen a state
         or the Q node value otherwise
       """
       return self.qvalues[(state, action)]

   def computeValueFromQValues(self, state):
       """
         Returns max_action Q(state,action)
         where the max is over legal actions.  Note that if
         there are no legal actions, which is the case at the
         terminal state, you should return a value of 0.0.
       """
       legal_actions = self.getLegalActions(state)
       if not legal_actions:
           return 0.0
       return max(self.getQValue(state, action) for action in legal_actions)

   def computeActionFromQValues(self, state):
       """
         Compute the best action to take in a state.  Note that if there
         are no legal actions, which is the case at the terminal state,
         you should return None.
       """
       legal_actions = self.getLegalActions(state)
       if not legal_actions:
           return None

       max_q_value = self.computeValueFromQValues(state)
       best_actions = [action for action in legal_actions if self.getQValue(state, action) == max_q_value]
       if not best_actions:
           # If all Q-values are equal, choose a random legal action
           return random.choice(legal_actions)
       else:
           return random.choice(best_actions)

   def getAction(self, state):
       """
         Compute the action to take in the current state.  With
         probability self.epsilon, we should take a random action and
         take the best policy action otherwise.  Note that if there are
         no legal actions, which is the case at the terminal state, you
         should choose None as the action.

         HINT: You might want to use util.flipCoin(prob)
         HINT: To pick randomly from a list, use random.choice(list)
       """
       legal_actions = self.getLegalActions(state)
       if not legal_actions:
           return None

       if util.flipCoin(self.epsilon):
           return random.choice(legal_actions)
       else:
           return self.getPolicy(state)

   def update(self, state, action, nextState, reward):
       """
         The parent class calls this to observe a
         state = action => nextState and reward transition.
         You should do your Q-Value update here

         NOTE: You should never call this function,
         it will be called on your behalf
       """
       next_value = self.computeValueFromQValues(nextState)
       td_error = reward + self.discount * next_value - self.getQValue(state, action)
       self.qvalues[(state, action)] += self.alpha * td_error

   def getPolicy(self, state):
       return self.computeActionFromQValues(state)

   def getValue(self, state):
       return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
   "Exactly the same as QLearningAgent, but with different default parameters"

   def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.02, numTraining=0, **args):
       """
       These default parameters can be changed from the pacman.py command line.
       For example, to change the exploration rate, try:
           python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

       alpha    - learning rate
       epsilon  - exploration rate
       gamma    - discount factor
       numTraining - number of training episodes, i.e. no learning after these many episodes
       """
       args['epsilon'] = epsilon
       args['gamma'] = gamma
       args['alpha'] = alpha
       args['numTraining'] = numTraining
       self.index = 0  # This is always Pacman
       QLearningAgent.__init__(self, **args)

   def getAction(self, state):
       """
       Simply calls the getAction method of QLearningAgent and then
       informs parent of action for Pacman.  Do not change or remove this
       method.
       """
       action = QLearningAgent.getAction(self, state)
       self.doAction(state, action)
       return action


class ApproximateQAgent(PacmanQAgent):
   """
      ApproximateQLearningAgent

      You should only have to overwrite getQValue
      and update.  All other QLearningAgent functions
      should work as is.
   """

   def __init__(self, extractor='IdentityExtractor', **args):
       self.featExtractor = util.lookup(extractor, globals())()
       PacmanQAgent.__init__(self, **args)
       self.weights = {
           "bias": 0.0,
           "closest-capsule": 0.5,
           "closest-ghost": -2.0,
           '#-of-ghosts-1-step-away': -1.0,
           'eats-food': 0.1,
           'closest-food': 0.1,
           "score": 1.0,
           "distance-to-ghost0": -0.5,
           "distance-to-ghost1": -0.5,
           "distance-to-ghost2": -0.5,
           "distance-to-ghost3": -0.5
       }

   def getWeights(self):
       return self.weights

   def getWeight(self, feature):
       return self.weights[feature]

   def getQValue(self, state, action):
       """
         Should return Q(state,action) = w * featureVector
         where * is the dotProduct operator
       """
       features = self.featExtractor.getFeatures(state, action)
       return sum(features[feature] * self.getWeight(feature) for feature in features)

   def update(self, state, action, nextState, reward):
       features = self.featExtractor.getFeatures(state, action)
       next_value = self.getValue(nextState)
       td_error = reward + self.discount * next_value - self.getQValue(state, action)

       for feature in features:
           self.weights[feature] += self.alpha * td_error * features[feature]


   def final(self, state):
       "Called at the end of each game."
       PacmanQAgent.final(self, state)

       if self.episodesSoFar == self.numTraining:
           pass