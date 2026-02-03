from typing import List, Dict, Optional, TypedDict, Literal

class GameState(TypedDict, total=False):
    """Global state of the game."""
    theme: str                  # Current theme (e.g., "Strong animals")
    history: List[str]          # Game log
    played_cards: List[int]     # List of cards played on the table (for validation)
    last_played_card: int       # The number of the last played card (if smaller than this, failure)
    utterances: Dict[str, str]  # {AgentID: "Spoken word"}
    turn_count: int             # Current turn number
    status: Literal["ACTIVE", "FAILED", "SUCCESS"]  # Game status
    deck: List[int]             # Remaining deck
    agents: List[str]           # List of agent IDs
    hands: Dict[str, int]       # {AgentID: CardNumber}
    votes: Dict[str, Literal["PLAY", "WAIT"]]  # {AgentID: "PLAY"|"WAIT"}
    finished_agents: List[str]  # Agents who already played
    speaker_reasonings: Dict[str, str]  # {AgentID: reasoning}
    estimator_thoughts: Dict[str, str]  # {AgentID: thought}
    debug: bool
    reveal: bool
    theme_override: str

class AgentState(TypedDict):
    """State of a single agent."""
    agent_id: str
    hand_card: int              # Secret number (1-100)
    word: Optional[str]         # The word the agent decided to say
