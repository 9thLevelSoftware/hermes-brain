"""The dream cycle — sleep-time consolidation (docs/design/learning-system.md §1).

The brain's background learning shift: while the user is idle (or on a
nightly schedule), a short-lived process consolidates episodic memory into
semantic patterns, mines Hermes's own outcome ledger to reweight what gets
recalled, resolves contradictions, and prunes what is no longer worth
keeping — then re-renders the lane-1 index.

Everything here runs OUT of the turn path, in a `hermes brain dream`
process (or the provider worker's bounded idle tick for the cheap
strategies). Exactly one dream runs at a time, enforced by the brain_lease
row (no fcntl — works on native Windows). Every mutating strategy ships
`dry_run` this release (ship-inert): it logs what it WOULD do to audit_log
and changes nothing until the user flips its mode to 'active'.
"""

from __future__ import annotations

from .run import run_dream

__all__ = ["run_dream"]
