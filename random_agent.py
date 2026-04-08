from agent import Agent
from oxono import Game
import random

class RandomAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
    
    def act(self, state, remaining_time):
        print(state)
        actions = list(Game.actions(state))
        print("Number of possible Actions: ", len(actions))
        print("player is ", state.current_player)
        return random.choice(actions)