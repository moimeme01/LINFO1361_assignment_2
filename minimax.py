from agent import Agent
from oxono import Game


class MinimaxAgent(Agent):
    def __init__(self, player, max_depth=1000):
        super().__init__(player)
        self.max_depth = max_depth

    def max_value(self, state, depth):

        if Game.is_terminal(state) or depth==0:
            return (Game.utility(state, self.player), None)
        
        v = -float('inf')
        move = None

        for a in Game.actions(state):
            inter_state = state.copy()
            Game.apply(inter_state, a)
            v2, _ = self.min_value(inter_state, depth-1)
            if v2 > v:
                v = v2
                move = a 
        return (v, move)

    
    def min_value(self, state, depth):

        if Game.is_terminal(state) or depth==0:
            return (Game.utility(state, self.player), None)
        
        v = float('inf')
        move = None 

        for a in Game.actions(state):
            inter_state = state.copy()
            Game.apply(inter_state, a)
            v2, _ = self.max_value(inter_state, depth-1)
            if v2 < v:
                v = v2
                move = a 
        return (v, move)



    def act(self, state, remaining_time):
        _, move = self.max_value(state, self.max_depth)
        return move