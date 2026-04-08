from oxono import Game, State
from agent import Agent
from math import *
import random
import time

class MCTSNode():
    def __init__(self, state, parent=None, action=None, player = None):
        """
        Définition d'un noeud dans l'arbre. Il contient initiallement les valeurs suivantes:

        state = état du plateau
        parent = Parent de l'état actuel. Initiallement None, ensuite état précédent.
        action = action ayant mené à cet état du jeu. 
        player = Joueur qui doit jouer cet état. (Pour la racine, on aura donc racine: player = 0, enfants A: player = 1, enfants B: player = 0)
        children = Liste des enfants de ce noeud.
        visits = Nombre de visites
        total_utility = Somme des utilités de ce noeud (+1 si la partie a mené à un gain du joueur player, -1 si perte, 0 si égalité)
        untried_actions = Liste d'actions pas encore visitées depuis ce noeud.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.player = player
        self.children = []
        self.visits = 0
        self.total_utility = 0
        self.untried_actions = Game.actions(state)
        
    def is_terminal(self):
        """
        check si le noeud est une feuille ou pas.
        """
        return Game.is_terminal(self.state)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self):
        best_child = None
        best_child_score = -float('inf')
        C = 0.9
        scores_list = []
        for child in self.children: 
            exploitation = child.total_utility / child.visits
            exploration = C*sqrt(log(self.visits)/child.visits)
            score = exploration + exploitation
            if score > best_child_score:
                best_child_score = score
                best_child = child
            scores_list.append(best_child_score)
        #print("All scores are: ", scores_list)
        #print("Best score found is: ", score)

        return (best_child, best_child_score, scores_list)
    
    def selection(self):
        # Si le noeud est une feuille, alors on s'arrete et on développe à partir de celui ci.
        if self.is_terminal():
            # print("Le noeud est une feuille, on retourne donc celui-ci.")
            return self
        
        # Si il n'est pas terminal, on regarde si il est fully expanded (pas trop compris)
        elif not self.is_fully_expanded():
            # print(f"Il reste encore {len(self.untried_actions)} à essayer donc on retourne self.")
            return self
        
        else: 
            # print("Le noeud n'est pas une feuille, est fully expanded, donc on prend le meilleur enfant et on récursionne.")
            best = self.best_child()
            # print(f"Le meilleur enfant est l'enfant à la position {self.children.index(best[0])} avec un score de {best[1]}")
            return best[0].selection()
        
    def expand(self):
        """
        La phase d'expension consiste à choisir un enfant au hasard et le rajouter aux enfants du parent qui est self.

        Donc on crée une copie de l'état parent auquel on applique l'action choisie. Ensuite on rentre les paramètres qui définissent un noeud de l'arbre, à savoir:
        état = nouvel état retourné par apply actions
        parent = self
        action = action choisie et appliquée
        player = joueur qui jouera à létape enfant.

        Ensuite on la rajoute dans les enfants du parents et on retourne l'enfant. 
        """
        choosed_action = random.choice(self.untried_actions)
        self.untried_actions.remove(choosed_action)
        copy_state = State.copy(self.state)
        Game.apply(copy_state, choosed_action)
        player = copy_state.current_player
        child = MCTSNode(state=copy_state, parent=self, action=choosed_action, player=player)
        self.children.append(child)
        return child
        

    def simulation(self):
        """
        On va effectuer une simulaton de l'enfant. On crée donc une copie de l'enfant afin de ne pas le modifier. Puis on applique des actions randoms tant que pas de victoires.
        Une fois qu'on est à un état final on doit retourner le joueur qui a gagné et donc l'utilité. On s'en fihce donc de l'état final. On le print pour débug uniquement 

        Etant donné que Game.utility retourne 1 si le joueur a gagné,  -1 si il a perdu et 0 si égalité, alors on peut retrouvé le winner avec cette fonction. 

        Par simplicité on va retourner l'etat final. Au moins on pourra lors de la back-propagation, pour chaque noeud qu'on remonte, faire:
        get player, utility de state et get_player, additionner cela 
        """
        actionlist = []
        simulation_state = State.copy(self.state)
        while not Game.is_terminal(simulation_state):
            action = random.choice(Game.actions(simulation_state))
            actionlist.append(action)
            Game.apply(simulation_state, action)
           
        if Game.is_terminal(simulation_state):
            pass
            # print(f"Utiity of player 0 (pink) is {Game.utility(simulation_state, 0)} and utility of player 1 (black) is: {Game.utility(simulation_state, 1)}")
        # Si on est ici c'est que l'état à rejoint un état final. On retourne cet état
        # print("Liste des actions ", actionlist)
        # print(simulation_state)
        if Game.utility(simulation_state, 0) == 1: 
            winner = 0
        elif Game.utility(simulation_state, 0) == -1:
            winner = 1
        else:
            winner = 2 # Si égalité on dit que les 2 sont gagnants
        return winner


    def backpropagate(self, winner: int):
        """
        On remonte l'arbre en modifiants: utilité totale et visites

        Une fois qu'on a un state_to_check égal à None, alors on est arrivé au dessus de la root, donc on a mis à jour tout l'arbre. 
        """
        state_to_check = self
        # print("State to check is the state: ", state_to_check.state)
        # print(f"and has {state_to_check.total_utility} as utility and {state_to_check.visits} visits")
        while state_to_check is not None:
            state_to_check.visits += 1
            player = state_to_check.player
            # print("Player to check the win is player: ", player)
            if winner == player: # Si le joueur qu'on check gagne => +1
                utility_state = 1 
            elif winner == 2: # Si le winner est 2, alors il y a égalité, donc utilité de 0
                utility_state = 0
            else: # Si en revanche il ne gagne pas ET PAS d'égalité alors il a perdu => -1
                utility_state = -1
            state_to_check.total_utility += utility_state
            # print("New utility of the state is: ", state_to_check.total_utility)
            # print("Visits of this state is: ", state_to_check.visits)
            # print(f"The node {state_to_check.state} have now {state_to_check.total_utility} as utility and {state_to_check.visits} visits")
            state_to_check = state_to_check.parent
            # print("To the parent...")

class MCTSTree():
    """
    Dans cette classe nous allons créé l'arbre initial, avec le noeud root. 

    Nous allons également réutiliser cette classe afin de transposer le noeud root de l'algoritme au noeud ou notre IA joue, en éliminant toute la partie supérieure. 
    Car admettons que le joueur opposé n'ait pas joué son meilleur coup, l'IA devra tout recalculer pour voir, sur base de ce coup, lequel est le meilleur.
    """
    def __init__(self, state):
        self.root = MCTSNode(state=State.copy(state), parent=None, action=None, player=state.current_player)
    
    def compare_states(self, state1, state2):
        if state1.current_player != state2.current_player:
            return False
        if state1.totem_O != state2.totem_O or state1.totem_X != state2.totem_X:
            return False
        if state1.pieces_x != state2.pieces_x or state1.pieces_o != state2.pieces_o:
            return False
        
        for line1, line2 in zip(state1.board, state2.board):
            for cell1, cell2 in zip(line1, line2):
                if cell1 != cell2:
                    return False                
        return True


    def re_root(self, state_to_find):
        """
        Dans cette fonction, nous allons changer le root de l'arbre dans l'état auquel le jeu est arrivé.
        Si l'état dans lequel on est arrivé est deja dans les enfants du root, alors on fait un mapping (on efface le parent et on met root à l'enfant), de cette
        manière la suite de l'arbre est conservée. Si non alors on crée un nouveau noeud dans l'arbre avec aucun, et comme action celle qui a mené à cet état. 
        """
        found = False
        for child in self.root.children:
            if self.compare_states(state_to_find, child.state):
                child.parent = None
                self.root = child
                found = True
                print("Founded, the root of the tree has been modified")
                break # Une fois qu'on a trouvé on peut arreter
        if not found:
            self.root = MCTSNode(state=state_to_find, parent=None, action=None, player=state_to_find.current_player)
        return self

class MCTSAgent(Agent):

    """
    Le but de l'agent va être dans un premier temps d'initialliser l'arbre pour ensuite retourner la meilleure action

    On va attribuer 10 secondes de recherche à l'algorithme. On verra oou ca nous mène. 

    (Nous allons mettre un paramètre de profondeur atteinte afin de voir combien de temps l'algorithme met pour calculer des noeuds et adapter le temps
    qu'on lui donne par coup.)

    """

    def __init__(self, player, search_time=40):
        super().__init__(player)
        self.search_time=search_time
        self.tree = None


    def print_elements(self):
        elements = []
        for child in self.tree.root.children:
            if child.visits == 0:
                score = 1000
            else:
                exploitation = child.total_utility / child.visits
                exploration = 1.4*sqrt(log(self.tree.root.visits)/child.visits)
                score = exploration + exploitation
            elements.append([child.action, score])
        print(elements)
        return(elements)

    def print_state(self, state):
        for elements in state.board:
            print(elements)
        print(f"Totems are at positions: {state.totem_O} and {state.totem_X}")
        print(f"The current player is {state.current_player}\n")
        return state


    def act(self, state, search_time):
        if self.tree is None:
            self.tree = MCTSTree(state)
            print("New tree created")
            print("Root is equal to ", self.tree.root.state)
            print("children are equal to ", self.tree.root.children)
            print("Is terminal? ", self.tree.root.is_terminal())
            print("Untried actions are: ", len(self.tree.root.untried_actions))
        else: 
            #print(f"Checking for the state {state} in the children of {self.tree.root.state}")
            self.tree = self.tree.re_root(state)
            #print("New tree root have been founded for the tree: ", self.tree)
        
            print("====================================")
            print(f"Analyse de la situation.\n")
            print("Pour ce parent:")
            self.print_state(self.tree.root.state)
            print(f"J'ai un maximum de {len(Game.actions(self.tree.root.state))} actions possibles.")
            print(f"Nous en avons deja visité: {len(self.tree.root.children)}. Pour l'instant le meilleur coup possible est: {self.tree.root.best_child()[0].action}")
            print(f"avec un score de {self.tree.root.best_child()[1]} avec une utilité totale de {self.tree.root.best_child()[0].total_utility} pour {self.tree.root.best_child()[0].visits} visites")
            print(f"Les autres coups ont comme valeurs:")
            for child in self.tree.root.children:
                exploitation = child.total_utility / child.visits
                exploration = 0.9*sqrt(log(self.tree.root.visits)/child.visits)
                score = exploration + exploitation
                print(f"Coup: {child.action}, score: {score}, visites: {child.visits}, utilité: {child.total_utility}.")
            print("====================================\n")

        
        depth = 0
        budget = self.search_time
        deadline = time.time() + budget
        print("calculating the best action for the state:")
        self.print_state(self.tree.root.state)

        while time.time() <= deadline:
            #print(f"****** Iteration number {num_iter} ******")
            selected = self.tree.root.selection()
            # print(f"Selection ...")
            # self.print_state(selected.state)

            # print("Expansion ...")
            child = selected.expand()
            # print("expanded state is:")
            # self.print_state(child.state)

            result = child.simulation()
            # print("Running simulation ...")
            # print("Simulation ended. Winner is: ", result)
            
            # print("Let's go for the backpropagation of the state.")
            child.backpropagate(result)

            # print("End of current iteration \n")
        #self.print_elements()
        #print(min(self.print_elements(), key=lambda x: x[1]))

        best_child = self.tree.root.best_child()
        

        print(f"Le meilleur enfant trouvé est l'état: {best_child[0].state} avec un score de {best_child[1]} avec une utilité totale de {best_child[0].total_utility} pour {best_child[0].visits} visites")
        
        self.last_action = best_child[0].action
        copy_state = State.copy(state)
        Game.apply(copy_state, best_child[0].action)
        self.tree.re_root(copy_state)

        return best_child[0].action
