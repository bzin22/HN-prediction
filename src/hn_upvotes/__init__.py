"""Predict Hacker News scores from information available at submission time.

Submodules are not imported here on purpose. ``embeddings``, ``models``, ``training``
and ``serving`` need optional dependencies (torch, fastapi), so importing them eagerly
would make ``import hn_upvotes`` fail in an environment that only has the core install.
"""

__version__ = "0.1.0"
