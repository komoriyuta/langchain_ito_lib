"""
Ito Game Graph - LangGraph-based implementation of Ito card game.

This module provides a LangGraph-based implementation that preserves all Ito game
functionality while allowing GRPO framework integration.
"""

from typing import Dict, List, Optional, Literal, Any
from langgraph.graph import StateGraph, END
from models.schemas import GameState
from utils.deck import create_deck, draw_card
from models.themes import THEMES_JA
import random


class ItoGameGraph:
    """
    LangGraph-based Ito game implementation.
    
    Provides full Ito functionality with LangGraph state management,
    making it suitable for both interactive play and GRPO training.
    """

    def __init__(
        self,
        agent_ids: List[str],
        human_agent_id: Optional[str] = None,
        theme: Optional[str] = None,
        max_turns: int = 20,
        debug: bool = False,
        reveal_hands: bool = False,
    ):
        self.agent_ids = agent_ids
        self.human_agent_id = human_agent_id
        self.theme = theme
        self.max_turns = max_turns
        self.debug = debug
        self.reveal_hands = reveal_hands

        # Import agents here to avoid circular imports
        from agents.speaker import generate_word as speaker_generate_word
        from agents.estimator import decide_action as estimator_decide_action
        from agents.discussion import (
            generate_question as discussion_generate_question,
            generate_player_question as discussion_generate_player_question,
            generate_answer as discussion_generate_answer,
        )

        self.speaker_generate_word = speaker_generate_word
        self.estimator_decide_action = estimator_decide_action
        self.discussion_generate_question = discussion_generate_question
        self.discussion_generate_player_question = discussion_generate_player_question
        self.discussion_generate_answer = discussion_generate_answer

        # Build graph
        self._graph = self._build_graph()
        self.app = self._graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GameState)

        # Add nodes
        workflow.add_node("setup", self._setup_node)
        workflow.add_node("speaking", self._speaking_node)
        workflow.add_node("voting", self._voting_node)
        workflow.add_node("execute_play", self._execute_play_node)
        workflow.add_node("wait_round", self._wait_round_node)

        # Set entry point
        workflow.set_entry_point("setup")

        # Add edges
        workflow.add_edge("setup", "speaking")
        workflow.add_edge("speaking", "voting")

        # Conditional edges
        workflow.add_conditional_edges(
            "voting",
            self._router_node,
            {
                "execute_play": "execute_play",
                "wait_round": "wait_round"
            }
        )

        workflow.add_conditional_edges(
            "execute_play",
            self._check_game_end,
            {
                "end": END,
                "voting": "voting"
            }
        )

        workflow.add_conditional_edges(
            "wait_round",
            self._check_game_end,
            {
                "end": END,
                "voting": "voting",
            },
        )

        return workflow

    def _setup_node(self, state: GameState) -> GameState:
        """Initialize the game."""
        if self.debug:
            print("--- Setup Node ---")

        # Use provided agent_ids if not in state
        agents = state.get("agents") or self.agent_ids

        # Keep pre-dealt hands when a complete hand mapping is provided.
        provided_hands = state.get("hands", {})
        if isinstance(provided_hands, dict) and set(provided_hands.keys()) == set(agents):
            hands = dict(provided_hands)
            deck = [card for card in create_deck() if card not in set(hands.values())]
        else:
            deck = create_deck()
            hands = {}
            for agent in agents:
                hands[agent] = draw_card(deck)

        # Select theme
        theme_override = (state.get("theme_override") or "").strip()
        theme = theme_override if theme_override else (self.theme or random.choice(THEMES_JA))

        initial_history = state.get("history")
        if isinstance(initial_history, list) and initial_history:
            history = list(initial_history)
        else:
            history = [f"ゲーム開始。お題: {theme}"]

        # Create new state with all fields
        new_state = dict(state)
        new_state.update({
            "theme": theme,
            "history": history,
            "played_cards": [],
            "last_played_card": 0,
            "utterances": {},
            "turn_count": 0,
            "status": "ACTIVE",
            "deck": deck,
            "agents": agents,
            "votes": {},
            "finished_agents": [],
            "speaker_reasonings": {},
            "estimator_thoughts": {},
            "debug": self.debug,
            "reveal": self.reveal_hands,
            "theme_override": theme_override,
            "hands": hands
        })

        if self.debug:
            print(f"Theme: {theme}")
            print(f"Agents: {agents}")
            if self.reveal_hands:
                print(f"Hands: {hands}")

        return new_state

    def _speaking_node(self, state: GameState) -> GameState:
        """All agents generate words based on their cards."""
        if self.debug:
            print("--- Speaking Node ---")
        
        theme = state["theme"]
        hands = state.get("hands", {})
        utterances = {}
        speaker_reasonings = dict(state.get("speaker_reasonings", {}))
        
        history_update = []
        history_text = "\n".join(state.get("history", []))
        
        for agent_id in state["agents"]:
            if agent_id in state.get("finished_agents", []):
                continue
            
            card = hands[agent_id]
            
            # Human agent input
            if agent_id == self.human_agent_id:
                print(f"\n=== {agent_id} の番 ===")
                print(f"お題: {theme}")
                print(f"あなたの数字（秘密）: {card}/100")
                word = input("数字を言わずに、度合いを表す単語/短いフレーズを入力してください: ").strip()
                if not word:
                    word = "（無言）"
                speaker_reasonings[agent_id] = ""
            else:
                # AI agent
                result = self.speaker_generate_word(theme, card, history=history_text)
                word = result.get("word", "Error")
                speaker_reasonings[agent_id] = str(result.get("reasoning", ""))

                if self.debug and self.reveal_hands:
                    print(f"{agent_id}: {word}  (数字={card})")
                else:
                    print(f"{agent_id}: {word}")

                if self.debug:
                    reasoning = speaker_reasonings.get(agent_id, "")
                    if reasoning:
                        print(f"  └ reason: {reasoning}")

            utterances[agent_id] = word
            history_update.append(f"{agent_id} の発言: 『{word}』")

        # Return updated state
        new_state = dict(state)
        new_state["utterances"] = utterances
        new_state["speaker_reasonings"] = speaker_reasonings
        new_state["history"] = state["history"] + history_update
        
        return new_state

    def _voting_node(self, state: GameState) -> GameState:
        """All agents decide whether to PLAY or WAIT."""
        if self.debug:
            print("--- Voting Node ---")
        
        hands = state.get("hands", {})
        utterances = state["utterances"]
        theme = state["theme"]
        last_played = state["last_played_card"]
        
        votes = {}
        estimator_thoughts = {}
        history_text = "\n".join(state.get("history", []))
        
        for agent_id in state["agents"]:
            if agent_id in state.get("finished_agents", []):
                continue
            
            card = hands[agent_id]
            word = utterances.get(agent_id, "")
            other_utterances = {k: v for k, v in utterances.items() if k != agent_id and k not in state.get("finished_agents", [])}
            
            # Human agent input
            if agent_id == self.human_agent_id:
                print(f"\n=== {agent_id} の判断 ===")
                print(f"場の最大値: {last_played}")
                if other_utterances:
                    print("他プレイヤーの発言:")
                    for k, v in other_utterances.items():
                        print(f"- {k}: {v}")
                else:
                    print("他プレイヤーの発言: (なし)")

                raw = input("今出すなら PLAY / 待つなら WAIT を入力: ").strip().upper()
                votes[agent_id] = "PLAY" if raw == "PLAY" else "WAIT"
                estimator_thoughts[agent_id] = ""
            else:
                # AI agent
                decision = self.estimator_decide_action(theme, last_played, other_utterances, card, word, history=history_text)
                votes[agent_id] = str(decision.get("action", "WAIT")).strip().upper()
                if votes[agent_id] not in {"PLAY", "WAIT"}:
                    votes[agent_id] = "WAIT"
                estimator_thoughts[agent_id] = str(decision.get("thought", ""))
                
                if self.debug:
                    thought = estimator_thoughts[agent_id]
                    if thought:
                        print(f"  └ thought({agent_id}): {thought}")
            
            print(f"{agent_id} の投票: {votes[agent_id]}")
        
        # Return updated state
        new_state = dict(state)
        new_state["votes"] = votes
        new_state["estimator_thoughts"] = estimator_thoughts
        
        return new_state

    def _router_node(self, state: GameState) -> Literal["execute_play", "wait_round"]:
        """Routes based on votes."""
        votes = state.get("votes", {})
        play_candidates = [agent for agent, action in votes.items() if action == "PLAY"]
        
        if play_candidates:
            return "execute_play"
        else:
            return "wait_round"

    def _execute_play_node(self, state: GameState) -> GameState:
        """The chosen agent plays their card."""
        if self.debug:
            print("--- Execute Play Node ---")
        
        votes = state.get("votes", {})
        hands = state.get("hands", {})
        play_candidates = [agent for agent, action in votes.items() if action == "PLAY"]
        
        # Tie-breaking: Smallest card plays first
        player = min(play_candidates, key=lambda x: hands[x])
        card = hands[player]
        
        print(f"!!! {player} がカードを出します: {card} !!!")
        
        new_played = state["played_cards"] + [card]
        finished = state.get("finished_agents", []) + [player]
        
        # Check for failure
        if card < state["last_played_card"]:
            status = "FAILED"
            print(f"ゲームオーバー: {card} は {state['last_played_card']} より小さい")
        elif len(finished) == len(state["agents"]):
            status = "SUCCESS"
            print("ゲームクリア！ すべて昇順で出せました。")
        else:
            status = "ACTIVE"
        
        # Return updated state
        new_state = dict(state)
        new_state["played_cards"] = new_played
        new_state["last_played_card"] = card
        new_state["finished_agents"] = finished
        new_state["history"] = state["history"] + [f"{player} が {card} を出した。"]
        new_state["status"] = status
        
        return new_state

    def _wait_round_node(self, state: GameState) -> GameState:
        """Everyone waits, enter discussion phase."""
        if self.debug:
            print("--- Wait Round Node ---")
        
        print("全員WAIT... 会話フェーズ（質問）に入ります。")

        theme = state["theme"]
        last_played = state["last_played_card"]
        utterances = state.get("utterances", {})
        history_text = "\n".join(state.get("history", []))

        # Get finished agents to exclude them
        finished_agents = state.get("finished_agents", [])
        active_agents = [a for a in state.get("agents", []) if a not in finished_agents]
        
        # Everyone proposes a question (avoid duplicates)
        proposals: dict[str, str] = {}
        seen_questions: set[str] = set()
        
        for agent_id in active_agents:
            my_word = str(utterances.get(agent_id, "")).strip()

            if agent_id == self.human_agent_id:
                q_in = input("あなたの質問（空ならスキップ）: ").strip()
                if q_in and q_in not in seen_questions:
                    proposals[agent_id] = q_in
                    seen_questions.add(q_in)
                    print(f"{agent_id} の質問: {q_in}")
                continue

            q_obj = self.discussion_generate_player_question(theme, last_played, utterances, my_word, history=history_text) or {}
            q_text = str(q_obj.get("question", "")).strip()
            if q_text and q_text not in seen_questions:
                proposals[agent_id] = q_text
                seen_questions.add(q_text)
                print(f"{agent_id} の質問: {q_text}")

        # If nobody proposed, fallback to moderator-style question
        if not proposals:
            q_obj = self.discussion_generate_question(theme, last_played, utterances, history=history_text) or {}
            q_text = str(q_obj.get("question", "")).strip()
            if q_text:
                proposals["(moderator)"] = q_text

        if not proposals:
            proposals["(moderator)"] = "それぞれの発言は、どれくらい強い/大きいイメージですか？"

        history_update = ["全員WAIT。"]

        # Everyone answers everyone's questions (only active agents answer)
        for q_owner, q in proposals.items():
            q = str(q).strip()
            if not q:
                continue
            print(f"\n質問（{q_owner}）: {q}")
            history_update.append(f"質問（{q_owner}）: {q}")

            for agent_id in active_agents:
                my_word = str(state.get("utterances", {}).get(agent_id, "")).strip()

                if agent_id == self.human_agent_id:
                    ans = input(f"あなたの回答（質問者={q_owner}）: ").strip()
                    if ans:
                        history_update.append(f"{agent_id} の回答（質問者={q_owner}）: {ans}")
                        print(f"{agent_id} の回答: {ans}")
                    continue

                a = self.discussion_generate_answer(theme, q, my_word, history=history_text).get("answer", "")
                a = str(a).strip()
                if a:
                    history_update.append(f"{agent_id} の回答（質問者={q_owner}）: {a}")
                    print(f"{agent_id} の回答: {a}")

        next_turn = state.get("turn_count", 0) + 1
        if next_turn >= self.max_turns:
            print("停滞が続いたためゲームを終了します。")
            new_state = dict(state)
            new_state["history"] = state["history"] + history_update + ["停滞が続いたため終了。"]
            new_state["turn_count"] = next_turn
            new_state["votes"] = {}
            new_state["status"] = "FAILED"
            return new_state

        # Return updated state
        new_state = dict(state)
        new_state["history"] = state["history"] + history_update
        new_state["turn_count"] = next_turn
        new_state["votes"] = {}
        
        return new_state

    def _check_game_end(self, state: GameState) -> Literal["end", "voting"]:
        """Check if game should end."""
        if state["status"] != "ACTIVE":
            return "end"
        return "voting"

    def run(self, initial_state: Optional[GameState] = None, verbose: bool = True) -> GameState:
        """
        Run the game to completion.
        
        Args:
            initial_state: Optional initial game state
            verbose: Whether to print node names during execution
            
        Returns:
            The final game state
        """
        # Build initial state
        if initial_state is None:
            initial_state = GameState(
                agents=self.agent_ids,
                debug=self.debug,
                reveal=self.reveal_hands,
                theme_override=self.theme or "",
            )
        
        # Run graph
        final_state = initial_state
        for output in self.app.stream(initial_state):
            # Each output contains the updated state after a node execution
            for key, value in output.items():
                if verbose and self.debug:
                    print(f"\n--- Node: {key} ---")
                
                # Update final_state with the latest state
                if value is not None:
                    final_state = value
                    
                    if value is not None and "history" in value:
                        if len(value["history"]) > 0:
                            if verbose or self.debug:
                                print(value["history"][-1])
                    
                    if value is not None and "status" in value:
                        if verbose or value["status"] != "ACTIVE":
                            print(f"Status: {value['status']}")
        
        return final_state

    def get_app(self):
        """Get the compiled LangGraph application for advanced usage."""
        return self.app


def create_game_graph(
    agent_ids: List[str],
    human_agent_id: Optional[str] = None,
    theme: Optional[str] = None,
    max_turns: int = 20,
    debug: bool = False,
    reveal_hands: bool = False,
) -> ItoGameGraph:
    """
    Convenience function to create an ItoGameGraph instance.
    
    Args:
        agent_ids: List of agent identifiers
        human_agent_id: Optional ID of human player for interactive mode
        theme: Optional theme override (random if not provided)
        max_turns: Maximum turns before game ends due to stagnation
        debug: Enable debug output
        reveal_hands: Show all players' hands (debugging)
        
    Returns:
        An ItoGameGraph instance
    """
    return ItoGameGraph(
        agent_ids=agent_ids,
        human_agent_id=human_agent_id,
        theme=theme,
        max_turns=max_turns,
        debug=debug,
        reveal_hands=reveal_hands,
    )


# Export public API
__all__ = [
    "ItoGameGraph",
    "create_game_graph",
]
