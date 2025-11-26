import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from .game_rules import GameState, get_legal_actions_3d, Coord

class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action # Action that led to this state
        
        self.children: Dict[Tuple[str, Coord], MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTSAgent:
    def __init__(self, model, num_simulations: int = 50, c_puct: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, root_state: GameState) -> Tuple[str, Coord]:
        root = MCTSNode(root_state)
        
        # Expand root first
        self._expand(root)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Select
            while node.children:
                node = self._select_child(node)
                
            # Expand & Evaluate
            # If terminal, value is reward. Else, use network.
            # For simplicity, we assume non-terminal during expansion unless max steps.
            value = self._expand(node)
            
            # Backpropagate
            self._backpropagate(node, value)
            
        # Select best action (most visited)
        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        return best_action
        
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float('inf')
        best_child = None
        
        for action, child in node.children.items():
            u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = child.value + u
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
        
    def _expand(self, node: MCTSNode) -> float:
        # 1. Get Legal Actions
        legal_actions = get_legal_actions_3d(node.state)
        
        if not legal_actions:
            return 0.0 # Terminal/Stuck
            
        # 2. Inference (Policy + Value)
        # Mock Inference if model is None (for testing)
        if self.model is None:
            policy_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
            value = 0.5
        else:
            # TODO: Implement actual tensor conversion and inference
            # For now, uniform prior
            policy_probs = {a: 1.0/len(legal_actions) for a in legal_actions}
            value = 0.5
            
        # 3. Create Children
        for action in legal_actions:
            if action not in node.children:
                # Apply action to get new state
                # This requires a state transition function.
                # We'll implement a lightweight one here or in game_rules.
                new_state = self._apply_action(node.state, action)
                node.children[action] = MCTSNode(new_state, parent=node, action=action, prior=policy_probs[action])
                
        return value
        
    def _backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            
    def _apply_action(self, state: GameState, action: Tuple[str, Coord]) -> GameState:
        type, coord = action
        new_tensor = state.tensor.copy()
        d, h, w = coord
        
        if type == 'ADD':
            new_tensor[0, d, h, w] = 1
        elif type == 'REMOVE':
            new_tensor[0, d, h, w] = 0
            
        # Check Phase Transition
        # If GROWTH and Connected -> REFINEMENT
        new_phase = state.phase
        if state.phase == 'GROWTH':
            # Check connectivity (Simplified: if we added something, check if it connects)
            # Full check is expensive.
            pass
            
        return GameState(tensor=new_tensor, phase=new_phase, step_count=state.step_count + 1)
