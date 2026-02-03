#!/usr/bin/env python3
"""
Example usage of Ito game library with LangGraph-based implementation.
"""

import os


def example_basic():
    """Basic usage with 3 AI agents."""
    print("\n=== Example 1: Basic AI-only game ===\n")
    from ito_graph import create_game_graph
    
    # Mock mode for demonstration (no API key needed)
    os.environ["ITO_FORCE_MOCK"] = "true"
    
    game = create_game_graph(
        agent_ids=["Alice", "Bob", "Charlie"],
        theme="動物の大きさ",
        debug=True,
        reveal_hands=True
    )
    result = game.run(verbose=True)
    
    print(f"\nFinal status: {result['status']}")
    print(f"Played cards: {result['played_cards']}")


def example_interactive():
    """Interactive game with human player."""
    print("\n=== Example 2: Interactive game ===\n")
    print("Note: You will be prompted for input during game.\n")
    from ito_graph import create_game_graph
    
    # Use real LLM for interactive experience (requires API key)
    # Set ITO_FORCE_MOCK=false or don't set it
    
    game = create_game_graph(
        agent_ids=["Human", "Alice", "Bob"],
        theme="生き物の強さ",
        human_agent_id="Human",
        max_turns=15
    )
    result = game.run(verbose=True)
    print(f"\nFinal status: {result['status']}")


def example_custom_initial_state():
    """Use custom initial state for fine-grained control."""
    print("\n=== Example 3: Custom initial state ===\n")
    from ito_graph import create_game_graph
    from models.schemas import GameState
    
    os.environ["ITO_FORCE_MOCK"] = "true"
    
    game = create_game_graph(
        agent_ids=["Player1", "Player2"],
        theme="食べ物の辛さ"
    )
    
    # You can provide a custom initial state
    custom_state = GameState(
        agents=["Player1", "Player2"],
        theme_override="食べ物の辛さ",
        debug=False
    )
    
    result = game.run(initial_state=custom_state, verbose=True)
    print(f"\nFinal status: {result['status']}")


def example_access_graph_app():
    """Access the LangGraph app directly for advanced usage."""
    print("\n=== Example 4: Access LangGraph app ===\n")
    from ito_graph import create_game_graph
    
    os.environ["ITO_FORCE_MOCK"] = "true"
    
    game = create_game_graph(
        agent_ids=["X", "Y", "Z"],
        theme="乗り物の速さ"
    )
    
    # Get the compiled LangGraph app
    app = game.get_app()
    
    # You can visualize the graph (requires graphviz)
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
        print("Graph visualization displayed")
    except ImportError:
        print("Install IPython and graphviz to visualize the graph")
    except Exception as e:
        print(f"Could not visualize graph: {e}")
    
    # Or just run the game
    result = game.run(verbose=True)
    print(f"\nFinal status: {result['status']}")


def example_history_analysis():
    """Demonstrate accessing and analyzing game history."""
    print("\n=== Example 5: History analysis ===\n")
    from ito_graph import create_game_graph
    
    os.environ["ITO_FORCE_MOCK"] = "true"
    
    game = create_game_graph(
        agent_ids=["Alpha", "Beta"],
        theme="場所の混み具合"
    )
    result = game.run(verbose=True)
    
    # Analyze the game
    print("\n--- Game Analysis ---")
    print(f"Status: {result['status']}")
    print(f"Turn count: {result['turn_count']}")
    print(f"Cards played: {len(result['played_cards'])}/{len(result['agents'])}")
    
    # Print the conversation history
    print("\n--- Conversation History ---")
    for i, entry in enumerate(result["history"], 1):
        print(f"{i}. {entry}")
    
    # Print final hands (for debugging)
    print(f"\n--- Final Hands ---")
    for agent, card in result["hands"].items():
        status = "PLAYED" if agent in result["finished_agents"] else "NOT PLAYED"
        print(f"{agent}: {card} ({status})")


if __name__ == "__main__":
    import sys
    
    examples = {
        "basic": example_basic,
        "interactive": example_interactive,
        "custom": example_custom_initial_state,
        "graph": example_access_graph_app,
        "history": example_history_analysis,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        for name in examples.keys():
            print(f"  python example.py {name}")
        print("\nRunning basic example by default...")
        example_basic()
