"""Hydration utilities for repository clones and PR file snapshots.

Imports are resolved lazily so light-weight callers can import the package
without importing git-heavy hydration modules until they are needed.
"""

__all__ = ["PRHydrator", "RepositoryHydrator"]


def __getattr__(name: str):
    """Resolve public hydration classes on first access."""
    if name == "PRHydrator":
        from curation.hydration.pr_hydrator import PRHydrator

        return PRHydrator
    if name == "RepositoryHydrator":
        from curation.hydration.repository_hydrator import RepositoryHydrator

        return RepositoryHydrator
    raise AttributeError(name)
