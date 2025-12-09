
import pytest
from alphabuilder.src.logic.mcts.node import MCTSNode
from alphabuilder.src.logic.mcts.engine import extract_batch_from_tree

def test_extract_batch_simple_depth1():
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
    actions, dist = extract_batch_from_tree(root, batch_size=2, max_depth=1)
    
    assert len(actions) == 2
    assert actions[0] == a1
    assert actions[1] == a3
    assert dist[a1] == 100
    assert dist[a3] == 80

def test_extract_batch_depth2():
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
    # PQ Step 1: [(100, a1), (20, a2)]
    # Pop a1 (100). Selected: [a1].
    # Push children of a1: [(60, b1), (30, b2), (20, a2)]
    # Pop b1 (60). Selected: [a1, b1].
    # Pop b2 (30). Selected: [a1, b1, b2].
    
    actions, _ = extract_batch_from_tree(root, batch_size=3, max_depth=2)
    
    assert len(actions) == 3
    assert actions[0] == a1
    assert actions[1] == b1
    assert actions[2] == b2

def test_extract_batch_max_depth_limit():
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
    
    # Max Depth 2. Should pick a1, b1. Should NOT pick c1 even if batch size allows.
    actions, _ = extract_batch_from_tree(root, batch_size=5, max_depth=2)
    
    assert len(actions) == 2
    assert a1 in actions
    assert b1 in actions
    assert c1 not in actions
