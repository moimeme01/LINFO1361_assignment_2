from __future__ import annotations
from oxono import Game, State
from agent import Agent
import time


_INF = float("inf")

# Score d'un alignement de N pièces dans une fenêtre de 4 (0..4)

_WIN    = [ 0,  3,  30,  300,  30_000 ]

# Bonus supplémentaire pour une menace à 3
_THREAT = 600

# Fraction du temps restant consacré par coup
_TIME_FRACTION = 0.06          # 6 % du temps restant
_TIME_MIN      = 0.25          # plancher 0.25 s
_TIME_MAX      = 6.0           # plafond  6 s

class MyAgent(Agent):
    """
    Agent Oxono basé sur Alpha-Beta + Iterative Deepening.

    Stratégie
    ---------
    · On joue toujours le coup qui maximise notre utilité en supposant que
      l'adversaire joue parfaitement (hypothèse minimax).
    · Alpha-Beta élimine les branches inutiles : en moyenne seule la racine
      carrée du nombre de noeuds bruts est visitée.
    · Iterative Deepening garantit qu'on a toujours une réponse disponible
      même si le temps s'épuise à mi-recherche.
    """

    def __init__(self, player: int):
        super().__init__(player)
        self.opp    = 1 - player
        self._t0    = 0.0
        self._limit = 0.0

    # interface

    def act(self, state: State, remaining_time: float) -> tuple:
        """Retourne la meilleure action dans le budget de temps imparti."""
        self._t0    = time.perf_counter()
        self._limit = max(_TIME_MIN, min(remaining_time * _TIME_FRACTION, _TIME_MAX))

        actions = Game.actions(state)
        if not actions:
            return None
        if len(actions) == 1:
            return actions[0]

        # coup gagnant ?
        for a in actions:
            s = state.copy(); Game.apply(s, a)
            if Game.is_terminal(s) and Game.utility(s, self.player) > 0:
                return a

        # itérative deepening
        best = self._order_root(state, actions)[0]   # fallback : meilleur selon heuristique

        for depth in range(1, 50):
            if self._time_up():
                break
            try:
                candidate = self._search(state, depth, actions)
                best = candidate            # on ne met à jour que si terminé
            except TimeoutError:
                break                       # on conserve le résultat du niveau précédent

        return best

    # recherche

    def _time_up(self) -> bool:
        return (time.perf_counter() - self._t0) >= self._limit

    # noeud racine
    def _search(self, state: State, depth: int, raw_actions: list) -> tuple:
        ordered = self._order_root(state, raw_actions)
        best, alpha = ordered[0], -_INF

        for a in ordered:
            if self._time_up():
                raise TimeoutError()
            s = state.copy(); Game.apply(s, a)
            v = self._ab(s, depth - 1, alpha, _INF)
            if v > alpha:
                alpha, best = v, a
            if alpha >= _INF / 2:          # victoire certaine trouvée
                break

        return best

    # alpha-beta récursif : retourne le score de la position
    def _ab(self, state: State, depth: int, α: float, β: float) -> float:
        if self._time_up():
            raise TimeoutError()

        # cas terminal
        if Game.is_terminal(state):
            return Game.utility(state, self.player) * 10_000

        if depth == 0:
            return self._eval(state)

        # ─ expansion
        actions = self._order_fast(state, Game.actions(state))

        if Game.to_move(state) == self.player:   # MAX
            v = -_INF
            for a in actions:
                s = state.copy(); Game.apply(s, a)
                v = max(v, self._ab(s, depth - 1, α, β))
                if v >= β:
                    return v          # coupure beta
                α = max(α, v)
        else:                                     # MIN
            v = _INF
            for a in actions:
                s = state.copy(); Game.apply(s, a)
                v = min(v, self._ab(s, depth - 1, α, β))
                if v <= α:
                    return v          # coupure alpha
                β = min(β, v)

        return v

    # ordonnancement

    def _order_root(self, state: State, actions: list) -> list:
        """
        Ordonnancement complet à la racine.
        Inclut une vérification de victoire immédiate et un calcul rapide
        de score pour classer le reste.
        """
        scored = []
        for a in actions:
            s = state.copy(); Game.apply(s, a)

            if Game.is_terminal(s):
                u = Game.utility(s, self.player)
                scored.append((u * 50_000, a))
            else:
                # Score heuristique léger (pas le eval complet) pour la racine
                scored.append((self._positional_score(state, a), a))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored]

    def _order_fast(self, state: State, actions: list) -> list:
        """
        Ordonnancement rapide sans copie d'état — utilisé aux noeuds internes.
        Critères : centralité + connectivité + totem bien placé.
        """
        board = state.board

        def key(a: tuple) -> float:
            _, totem_dst, (r, c) = a

            # 1. Centralité de la pièce posée (0..10)
            ctr = (5 - abs(2*r - 5)) + (5 - abs(2*c - 5))

            # 2. Adjacence à des pièces existantes (connectivité)
            adj_own = adj_opp = 0
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < 6 and 0 <= nc < 6:
                    cell = board[nr][nc]
                    if cell:
                        if cell[1] == self.player:
                            adj_own += 1
                        else:
                            adj_opp += 1

            # 3. Centralité du totem de destination
            tr, tc = totem_dst
            totem_ctr = (5 - abs(2*tr - 5)) + (5 - abs(2*tc - 5))

            return ctr + 3*adj_own + 4*adj_opp + 0.3*totem_ctr

        return sorted(actions, key=key, reverse=True)

    def _positional_score(self, state: State, action: tuple) -> float:
        """Score heuristique d'un coup à la racine (ne copie pas l'état)."""
        _, _, (r, c) = action
        board = state.board

        ctr = (5 - abs(2*r - 5)) + (5 - abs(2*c - 5))
        adj_own = adj_opp = 0
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < 6 and 0 <= nc < 6:
                cell = board[nr][nc]
                if cell:
                    if cell[1] == self.player:
                        adj_own += 1
                    else:
                        adj_opp += 1
        return ctr + 3*adj_own + 4*adj_opp

    # évaluation

    def _eval(self, state: State) -> float:
        """
        Fonction d'évaluation statique du plateau.

        Score = Σ (fenêtres horizontales + verticales de taille 4)
        Chaque fenêtre est évaluée sur deux axes indépendants :
          · alignement COULEUR  (toutes les pièces du même joueur)
          · alignement SYMBOLE  ('x' ou 'o') — partagé entre joueurs
        """
        b = state.board
        s = 0.0

        # fenêtres horizontales
        for r in range(6):
            for c in range(3):
                s += self._score_window(b[r][c], b[r][c+1], b[r][c+2], b[r][c+3])

        # fenêtres verticales
        for c in range(6):
            for r in range(3):
                s += self._score_window(b[r][c], b[r+1][c], b[r+2][c], b[r+3][c])

        return s

    def _score_window(self, c0, c1, c2, c3) -> float:
        """
        Évalue une fenêtre de 4 cellules (None ou (symbole, joueur)).

        Règles :
          · Une fenêtre « contestée » (mix des deux joueurs) pour un axe
            donné est neutralisée (score 0 sur cet axe).
          · Plus le nombre de pièces dans une fenêtre non contestée est
            grand, plus le score est élevé (progression géométrique).
          · Une menace à 3 reçoit un bonus supplémentaire (_THREAT) car
            elle doit être bloquée immédiatement.
          · Les alignements symbole sont pondérés à 0.6× (moins décisifs
            car partagés entre les deux joueurs).
        """
        cells = (c0, c1, c2, c3)
        s = 0.0

        # Couleur
        my_c = sum(1 for x in cells if x and x[1] == self.player)
        op_c = sum(1 for x in cells if x and x[1] == self.opp)

        if op_c == 0 and my_c:          # fenêtre pure pour moi
            s += _WIN[my_c]
            if my_c == 3:
                s += _THREAT            # menace à bloquer par l'adversaire
        elif my_c == 0 and op_c:        # fenêtre pure pour l'adversaire
            s -= _WIN[op_c]
            if op_c == 3:
                s -= _THREAT            # je dois bloquer ça

        # Symboles
        for sym in ('x', 'o'):
            n_sym  = sum(1 for x in cells if x and x[0] == sym)
            n_oth  = sum(1 for x in cells if x and x[0] != sym)   # autre symbole

            if n_oth == 0 and n_sym > 1:    # fenêtre pure pour ce symbole
                # Qui profite le plus de cet alignement ?
                my_s = sum(1 for x in cells if x and x[0] == sym and x[1] == self.player)
                op_s = n_sym - my_s
                sign = 1 if my_s > op_s else (-1 if op_s > my_s else 0)

                s += sign * (_WIN[n_sym] * 0.6 + (_THREAT * 0.6 if n_sym == 3 else 0))

        return s


if __name__ == "__main__":
    import random

    class _RandomAgent(Agent):
        def act(self, state, remaining_time):
            return random.choice(Game.actions(state))

    print("Test : MyAgent (joueur 0) vs RandomAgent (joueur 1)")
    wins, losses, draws = 0, 0, 0

    for game_i in range(10):
        state = State()
        times = [300.0, 300.0]
        agents = [MyAgent(0), _RandomAgent(1)]

        while not Game.is_terminal(state):
            p = Game.to_move(state)
            t0 = time.perf_counter()
            action = agents[p].act(state.copy(), times[p])
            times[p] -= (time.perf_counter() - t0)
            Game.apply(state, action)

        u = Game.utility(state, 0)
        if u == 1:   wins   += 1
        elif u == -1: losses += 1
        else:         draws  += 1

    print(f"  Victoires: {wins}/10  Défaites: {losses}/10  Nuls: {draws}/10")