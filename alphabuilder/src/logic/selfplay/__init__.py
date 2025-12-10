"""
Self-Play Module for AlphaBuilder MCTS.

Provides storage and utilities for self-play game data.
"""

from .storage import (
    # Data classes
    GameInfo,
    GameStep,
    GameSummary,
    SelectedAction,
    MCTSStats,
    # Database functions
    initialize_selfplay_db,
    save_game,
    record_step,
    update_game_final,
    load_game,
    load_game_step,
    load_all_game_steps,
    list_games,
    get_game_count,
    get_last_step,
    get_incomplete_games,
    generate_game_id,
)

from .reward import (
    # Constants
    MU_SCORE,
    SIGMA_SCORE,
    ALPHA_VOL,
    # Core functions
    calculate_raw_score,
    calculate_reward,
    # Phase-specific
    get_phase1_reward,
    get_phase2_terminal_reward,
    check_structure_connectivity,
    estimate_reward_components,
    # Island analysis
    analyze_structure_islands,
    calculate_island_penalty,
    get_reward_with_island_penalty,
    calculate_connectivity_reward,
)

__all__ = [
    'GameInfo',
    'GameStep',
    'GameSummary',
    'SelectedAction',
    'MCTSStats',
    'initialize_selfplay_db',
    'save_game',
    'record_step',
    'update_game_final',
    'load_game',
    'load_game_step',
    'load_all_game_steps',
    'list_games',
    'get_game_count',
    'get_last_step',
    'get_incomplete_games',
    'generate_game_id',
    'calculate_connectivity_reward',
]

