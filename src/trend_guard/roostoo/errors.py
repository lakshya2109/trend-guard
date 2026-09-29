"""Exceptions raised by the Roostoo client. See BUILD_SPEC.md Section 4."""

from __future__ import annotations


class RoostooAPIError(Exception):
    """A request failed after all retries, or the transport itself failed unrecoverably."""


class OrderStateUnknownError(Exception):
    """An order-creating POST timed out or errored before we could read the response.

    The order may or may not have been accepted server-side. Never resubmit blindly: the caller
    must reconcile via query_order before taking any further action on this intent.
    """


class ShortsDisabledError(Exception):
    """The competition account does not have shorting enabled.

    The engine must catch this once, permanently disable the short sleeve for the session, and
    log it (BUILD_SPEC.md Section 4.2).
    """
