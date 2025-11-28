import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from alphabuilder.src.neural.dataset import PhysicsAugment, AlphaBuilderDataset
from alphabuilder.src.neural.trainer import AlphaLoss

class TestPhysicsAugment:
    @pytest.fixture
    def sample_data(self):
        # Shape: (5, 4, 4, 4) -> (C, D, H, W)
        # Channels: 0:Rho, 1:Mask, 2:Fx, 3:Fy, 4:Fz
        state = torch.zeros((5, 4, 4, 4), dtype=torch.float32)
        policy = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
        
        # Set specific values to track flips
        # Voxel at (0, 0, 0)
        state[0, 0, 0, 0] = 1.0 # Density
        state[2, 0, 0, 0] = 10.0 # Fx
        state[3, 0, 0, 0] = 20.0 # Fy
        state[4, 0, 0, 0] = 30.0 # Fz
        
        # Policy at (0, 0, 0)
        policy[0, 0, 0, 0] = 1.0 # Add
        
        return state, policy

    def test_flip_x_inverts_force_x(self, sample_data):
        state, policy = sample_data
        augment = PhysicsAugment()
        
        # Force Flip X (Axis 1) by mocking random
        # Sequence: Flip X (0.9), Skip Y (0.1), Skip Z (0.1)
        with patch('random.random', side_effect=[0.9, 0.1, 0.1]): 
            aug_state, aug_policy = augment(state.clone(), policy.clone())
            
        # Check Geometry Flip on Axis 1 (Depth)
        # Original (0,0,0) should move to (3,0,0) if D=4
        
        # Check Density moved
        assert aug_state[0, 3, 0, 0] == 1.0
        assert aug_state[0, 0, 0, 0] == 0.0
        
        # Check Force X Inversion
        # Original Fx=10.0. New Fx should be -10.0 at (3,0,0)
        assert aug_state[2, 3, 0, 0] == -10.0
        
        # Check Force Y and Z Unchanged (sign-wise)
        assert aug_state[3, 3, 0, 0] == 20.0
        assert aug_state[4, 3, 0, 0] == 30.0

    def test_flip_y_inverts_force_y(self, sample_data):
        state, policy = sample_data
        augment = PhysicsAugment()
        
        # Force Flip Y (Axis 2)
        # Sequence: Skip X (0.1), Flip Y (0.9), Skip Z (0.1)
        with patch('random.random', side_effect=[0.1, 0.9, 0.1]): 
            aug_state, aug_policy = augment(state.clone(), policy.clone())
            
        # Check Geometry Flip on Axis 2 (Height)
        # Original (0,0,0) -> (0,3,0)
        assert aug_state[0, 0, 3, 0] == 1.0
        
        # Check Force Y Inversion
        assert aug_state[3, 0, 3, 0] == -20.0
        
        # Check Force X and Z Unchanged
        assert aug_state[2, 0, 3, 0] == 10.0
        assert aug_state[4, 0, 3, 0] == 30.0

    def test_flip_z_inverts_force_z(self, sample_data):
        state, policy = sample_data
        augment = PhysicsAugment()
        
        # Force Flip Z (Axis 3)
        # Sequence: Skip X (0.1), Skip Y (0.1), Flip Z (0.9)
        with patch('random.random', side_effect=[0.1, 0.1, 0.9]): 
            aug_state, aug_policy = augment(state.clone(), policy.clone())
            
        # Check Geometry Flip on Axis 3 (Width)
        # Original (0,0,0) -> (0,0,3)
        assert aug_state[0, 0, 0, 3] == 1.0
        
        # Check Force Z Inversion
        assert aug_state[4, 0, 0, 3] == -30.0
        
        # Check Force X and Y Unchanged
        assert aug_state[2, 0, 0, 3] == 10.0
        assert aug_state[3, 0, 0, 3] == 20.0

class TestAlphaBuilderDataset:
    @patch('alphabuilder.src.neural.dataset.sqlite3')
    @patch('alphabuilder.src.neural.dataset.pickle')
    def test_getitem_normalization(self, mock_pickle, mock_sqlite):
        # Setup Mock DB
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock Data
        mock_state = np.zeros((5, 10, 10, 10), dtype=np.float32)
        mock_policy = np.zeros((2, 10, 10, 10), dtype=np.float32)
        mock_fitness = 100.0 # High compliance (bad) or Low compliance (good)? 
        # Spec says: "Target V = -log(Compliance Final)" if fitness is compliance.
        # Code says: value_t = torch.log(torch.tensor([fitness]) + epsilon)
        # Wait, the code implemented `torch.log(fitness)`. 
        # If fitness in DB is Compliance (e.g. 5000), then log(5000) ~ 8.5.
        # If fitness in DB is 1/Compliance (e.g. 0.0002), then log(0.0002) ~ -8.5.
        # Let's verify what the code actually does.
        
        mock_pickle.loads.side_effect = [mock_state, mock_policy]
        mock_cursor.fetchone.return_value = (b'state', b'policy', mock_fitness)
        
        # Init Dataset
        dataset = AlphaBuilderDataset("dummy.db", augment=False)
        dataset.indices = [1] # Fake index
        
        # Call __getitem__
        state, policy, value = dataset[0]
        
        # Verify Value Normalization
        # Code: torch.log(tensor(fitness) + 1e-6)
        expected_value = np.log(mock_fitness + 1e-6)
        assert torch.isclose(value, torch.tensor([expected_value], dtype=torch.float32), atol=1e-5)

    @patch('alphabuilder.src.neural.dataset.sqlite3')
    @patch('alphabuilder.src.neural.dataset.pickle')
    def test_augmentation_flag(self, mock_pickle, mock_sqlite):
        # Setup Mock
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        mock_state = np.zeros((5, 10, 10, 10), dtype=np.float32)
        mock_policy = np.zeros((2, 10, 10, 10), dtype=np.float32)
        mock_fitness = 1.0
        
        mock_pickle.loads.side_effect = [mock_state, mock_policy]
        mock_cursor.fetchone.return_value = (b'state', b'policy', mock_fitness)
        
        # Init Dataset with augment=True
        dataset = AlphaBuilderDataset("dummy.db", augment=True)
        dataset.indices = [1]
        
        # Mock the augmentor
        dataset.augmentor = MagicMock(return_value=(torch.tensor(mock_state), torch.tensor(mock_policy)))
        
        # Call __getitem__
        dataset[0]
        
        # Verify augmentor was called
        dataset.augmentor.assert_called_once()

class TestAlphaLoss:
    def test_loss_computation(self):
        loss_fn = AlphaLoss(bce_weight=1.0, dice_weight=1.0)
        
        # Pred: Logits. Target: Binary Mask.
        # Case 1: Perfect Match
        # Target = 1, Logits = 100 (Sigmoid ~ 1)
        # Target = 0, Logits = -100 (Sigmoid ~ 0)
        target = torch.tensor([[[1.0, 0.0]]])
        logits = torch.tensor([[[10.0, -10.0]]])
        
        loss = loss_fn(logits, target)
        
        # BCE should be near 0
        # Dice should be near 0 (Score ~ 1)
        assert loss.item() < 0.1
        
    def test_loss_mismatch(self):
        loss_fn = AlphaLoss(bce_weight=1.0, dice_weight=1.0)
        
        # Case 2: Complete Mismatch
        target = torch.tensor([[[1.0, 0.0]]])
        logits = torch.tensor([[[-10.0, 10.0]]]) # Predict 0 for 1, 1 for 0
        
        loss = loss_fn(logits, target)
        
        # BCE should be high
        # Dice should be high (Score ~ 0)
        assert loss.item() > 1.0
