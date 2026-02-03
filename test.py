#!/usr/bin/env python3
"""
Test script to verify Ito game implementation works correctly.
"""

import os
import sys

# Force mock mode for testing
os.environ["ITO_FORCE_MOCK"] = "true"


def test_basic_ai_game():
    """Test basic AI-only game."""
    print("\n=== Test 1: Basic AI-only game (mock mode) ===\n")
    from ito_graph import create_game_graph
    
    game = create_game_graph(
        agent_ids=["Alice", "Bob", "Charlie"],
        theme="動物の大きさ",
        debug=True,
        reveal_hands=True
    )
    
    result = game.run(verbose=True)
    
    # Verify game ended
    assert result["status"] in ["SUCCESS", "FAILED", "ACTIVE"], f"Invalid status: {result['status']}"
    print(f"\n✓ Test 1 passed! Status: {result['status']}")
    
    # Check that all agents have cards
    assert "hands" in result, "State should contain hands"
    assert len(result["hands"]) == 3, f"Should have 3 hands, got {len(result['hands'])}"
    
    # Check that history is populated
    assert len(result["history"]) > 0, "History should not be empty"
    
    print(f"✓ History entries: {len(result['history'])}")
    print(f"✓ Hands: {result['hands']}")
    
    return True


def test_small_game():
    """Test with 2 agents."""
    print("\n=== Test 2: Small game with 2 agents ===\n")
    from ito_graph import create_game_graph
    
    game = create_game_graph(
        agent_ids=["Agent1", "Agent2"],
        theme="食べ物の辛さ",
        max_turns=5,  # Short game for quick test
        debug=False
    )
    
    result = game.run(verbose=False)
    
    assert result["status"] in ["SUCCESS", "FAILED"], f"Invalid status: {result['status']}"
    print(f"✓ Test 2 passed! Status: {result['status']}")
    
    return True


def test_get_app():
    """Test that we can access the LangGraph app."""
    print("\n=== Test 3: Access LangGraph app ===\n")
    from ito_graph import create_game_graph
    
    game = create_game_graph(
        agent_ids=["Alice", "Bob"],
        theme="道具の便利さ"
    )
    
    app = game.get_app()
    assert app is not None, "App should not be None"
    
    print("✓ Test 3 passed! App is accessible")
    
    return True


def test_initial_state_override():
    """Test that we can provide initial state."""
    print("\n=== Test 4: Initial state override ===\n")
    from ito_graph import create_game_graph
    from models.schemas import GameState
    
    game = create_game_graph(
        agent_ids=["X", "Y"],
        theme="乗り物の速さ"
    )
    
    custom_state = GameState(
        agents=["X", "Y"],
        theme_override="乗り物の速さ",
        debug=True
    )
    
    result = game.run(initial_state=custom_state, verbose=False)
    
    assert result["status"] in ["SUCCESS", "FAILED"], f"Invalid status: {result['status']}"
    print(f"✓ Test 4 passed! Status: {result['status']}")
    
    return True


def test_game_state_complete():
    """Test that final state has all required fields."""
    print("\n=== Test 5: Complete game state ===\n")
    from ito_graph import create_game_graph
    
    game = create_game_graph(
        agent_ids=["Player1", "Player2"],
        theme="キャラの強さ（RPG）",
        max_turns=3
    )
    
    result = game.run(verbose=False)
    
    # Check all required fields
    required_fields = [
        "theme", "history", "played_cards", "last_played_card",
        "utterances", "turn_count", "status", "agents", "hands",
        "votes", "finished_agents", "speaker_reasonings",
        "estimator_thoughts"
    ]
    
    for field in required_fields:
        assert field in result, f"Missing field: {field}"
    
    print(f"✓ Test 5 passed! All {len(required_fields)} required fields present")
    
    return True


def main():
    """Run all tests."""
    tests = [
        test_basic_ai_game,
        test_small_game,
        test_get_app,
        test_initial_state_override,
        test_game_state_complete,
    ]
    
    print("=" * 60)
    print("Ito Game - Test Suite")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test error: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
