
import pytest
from alphabuilder.src.logic.mcts.node import MCTSNode
from alphabuilder.src.logic.mcts.engine import select_pv_sequences

def test_select_pv_sequences_simple():
    root = MCTSNode(prior=1.0)
    
    # Children of root
    a1 = (0, 0, 0, 1)
    a2 = (0, 0, 0, 2)
    a3 = (0, 0, 0, 3)
    
    root.children[a1] = MCTSNode(prior=0.3)
    root.children[a1].visit_count = 100
    
    root.children[a2] = MCTSNode(prior=0.3)
    root.children[a2].visit_count = 50
    
    root.children[a3] = MCTSNode(prior=0.4)
    root.children[a3].visit_count = 80
    
    # Batch 2 -> expected [a1, a3] (100, 80)
    # Note: select_pv_sequences sorts by visits
    actions = select_pv_sequences(root, max_actions=2, max_depth=1)
    
    assert len(actions) == 2
    assert actions[0] == a1
    assert actions[1] == a3

def test_select_pv_sequences_depth():
    root = MCTSNode(prior=1.0)
    
    # Root Children
    a1 = (0, 0, 0, 1) # V=100
    a2 = (0, 0, 0, 2) # V=20
    
    root.children[a1] = MCTSNode(prior=0.5)
    root.children[a1].visit_count = 100
    root.children[a1].is_expanded = True
    
    root.children[a2] = MCTSNode(prior=0.5)
    root.children[a2].visit_count = 20
    
    # Children of a1
    b1 = (0, 0, 0, 11) # V=60 (child of a1)
    b2 = (0, 0, 0, 12) # V=30 (child of a1)
    
    root.children[a1].children[b1] = MCTSNode(prior=0.5)
    root.children[a1].children[b1].visit_count = 60
    
    root.children[a1].children[b2] = MCTSNode(prior=0.5)
    root.children[a1].children[b2].visit_count = 30
    
    # Request Batch 3, Depth 2
    # Logic:
    # 1. Start with best child a1 (100 visits)
    # 2. Add a1
    # 3. Recurse to a1 children: b1 (60), b2 (30)
    # 4. Add b1 (best child of a1)
    # 5. Add b2 (second best child of a1 if depth/actions allow? No, implementation is greedy PV)
    # Let's verify logic: 
    # select_pv_sequences iterates over Root children.
    # a. Child a1. Add a1. Recurse a1. Child b1. Add b1. (Depth 2 reached). Sequence [a1, b1].
    # b. Child a2. Add a2. (Depth 1 reached). Sequence [a2].
    # Batch limit 3. Total actions: a1, b1, a2.
    
    actions = select_pv_sequences(root, max_actions=3, max_depth=2)
    
    assert len(actions) == 3
    assert a1 in actions
    assert b1 in actions
    assert a2 in actions
    # Order depends on implementation details, but usually breadth roots come first or interleaved?
    # Actually, the doc says "For each top child... follow best path".
    # So it should be a1, b1, then a2.
    
    assert actions[0] == a1
    assert actions[1] == b1
    assert actions[2] == a2

def test_select_pv_sequences_depth_limit():
    root = MCTSNode(prior=1.0)
    a1 = (0, 0, 0, 1)
    root.children[a1] = MCTSNode(prior=1.0)
    root.children[a1].visit_count = 100
    root.children[a1].is_expanded = True
    
    b1 = (0, 0, 0, 2)
    root.children[a1].children[b1] = MCTSNode(prior=1.0)
    root.children[a1].children[b1].visit_count = 90
    root.children[a1].children[b1].is_expanded = True
    
    c1 = (0, 0, 0, 3)
    root.children[a1].children[b1].children[c1] = MCTSNode(prior=1.0)
    root.children[a1].children[b1].children[c1].visit_count = 80
    
    # Max Depth 2. 
    # Path: a1 -> b1 -> c1 (depth 3)
    # Should select a1, b1. c1 is too deep relative to recursion start? 
    # Actually max_depth is usually relative to root.
    # If max_depth=2, we get a1 (depth 1), b1 (depth 2). c1 (depth 3) excluded.
    
    actions = select_pv_sequences(root, max_actions=5, max_depth=2)
    
    assert len(actions) == 2
    assert a1 in actions
    assert b1 in actions
    assert c1 not in actions

