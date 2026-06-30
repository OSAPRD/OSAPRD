"""
GitHub token rotation with lightweight rate-limit tracking.

Discovery and enrichment both call GitHub APIs and share the same token
semantics: use the active token until it is invalid or exhausted, then rotate to
the next configured token. This helper centralizes that behavior so scrapers do
not each implement subtly different retry decisions.
"""

import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TokenInfo:
    """Runtime state for one GitHub token."""

    token: str
    remaining: Optional[int] = None
    reset: Optional[int] = None


class TokenManager:
    """Round-robin token manager with rate-limit awareness."""

    def __init__(self, tokens: List[str]):
        """Initialize token rotation state from non-empty token input."""
        if not tokens:
            raise ValueError("At least one token is required")
        self.tokens_info = [TokenInfo(token=t) for t in tokens]
        self.current_index = 0
        self.invalid_tokens: set[str] = set()

    def get_token(self) -> str:
        """Return a usable token, rotating past invalid or currently exhausted ones."""
        for _ in range(len(self.tokens_info)):
            token_info = self.tokens_info[self.current_index]
            if token_info.token in self.invalid_tokens:
                self.rotate_token()
                continue
            now = time.time()
            if token_info.remaining == 0 and token_info.reset and now < token_info.reset:
                self.rotate_token()
                continue
            return token_info.token
        raise RuntimeError("All tokens are invalid or exhausted.")

    def rotate_token(self):
        """Move to the next token that is not invalid and can currently be used."""
        for _ in range(len(self.tokens_info)):
            self.current_index = (self.current_index + 1) % len(self.tokens_info)
            token_info = self.tokens_info[self.current_index]
            if token_info.token in self.invalid_tokens:
                continue
            now = time.time()
            if (token_info.remaining is None or token_info.remaining > 0) or (
                token_info.reset and int(now) > token_info.reset
            ):
                print(f"Switched to token index {self.current_index}")
                return
        raise RuntimeError("All tokens are exhausted until reset!")

    def invalidate_current(self) -> None:
        """Mark the current token as rejected by GitHub and rotate away."""
        token = self.tokens_info[self.current_index].token
        self.invalid_tokens.add(token)
        print(f"Marked token index {self.current_index} as invalid.")
        if len(self.invalid_tokens) >= len(self.tokens_info):
            raise RuntimeError("All tokens are invalid.")
        self.rotate_token()

    def update_limit(self, remaining: int, reset_timestamp: int):
        """Update rate-limit state for the token used by the latest request."""
        self.tokens_info[self.current_index].remaining = remaining
        self.tokens_info[self.current_index].reset = reset_timestamp

    def get_all_reset_times(self) -> List[Optional[int]]:
        """Return reset timestamps for diagnostics or wait-time calculation."""
        return [token_info.reset for token_info in self.tokens_info]

    def update_index(self, index: int):
        """Set the active token index after external validation or resume."""
        if 0 <= index < len(self.tokens_info):
            self.current_index = index
        else:
            raise IndexError("Token index out of range")
