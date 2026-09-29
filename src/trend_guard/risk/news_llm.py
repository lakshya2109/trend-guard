"""Optional LLM news veto overlay. BUILD_SPEC.md Section 9. Milestone 7, optional, off by default
(`risk_cfg.news_llm.enabled` is False in the committed config).

Only the deterministic mapping from a parsed LLM classification to a risk action is implemented here,
and it is pure (no I/O). The live feedparser polling + anthropic API call are intentionally deferred.
"""

from __future__ import annotations

from trend_guard.config import RiskConfig

_SEVERE = 3
_MARKET_CONFIDENCE_MIN = 0.7
_HACK_CATEGORIES = {"hack", "exploit", "delisting"}
_MARKET_MULT = 0.5
_MARKET_DURATION_H = 6
_COIN_BLOCK_H = 24


def apply_news_action(parsed: dict, risk_cfg: RiskConfig) -> dict:
    """Deterministic mapping from the LLM's parsed classification to a risk action. BUILD_SPEC Section 9.

    `parsed` is the LLM's parsed JSON: {"scope": "market"|"coin"|"none", "coins": [...],
    "severity": 0-3, "category": ..., "confidence": 0-1, "rationale": ...}.

    Rules (BUILD_SPEC Section 9):
      - scope == "market", severity == 3, confidence >= 0.7: news_mult 0.5 on all crypto for 6h.
      - scope == "coin", severity == 3, category in {hack, exploit, delisting}: target 0 for that
        coin and block entries for 24h.
      - anything else (including when news_llm is disabled): no-op, news_mult 1.0.

    If `risk_cfg.news_llm.enabled` is False, always returns the no-op action -- this function can
    still be called and tested independently of the enabled flag, but the engine should not act on
    a non-no-op result while overlay is disabled.
    """
    no_op = {"news_mult": 1.0}
    if not risk_cfg.news_llm.enabled:
        return no_op

    scope = parsed.get("scope")
    severity = parsed.get("severity")
    confidence = parsed.get("confidence")
    category = parsed.get("category")
    coins = parsed.get("coins") or []

    market_severe = scope == "market" and severity == _SEVERE
    if market_severe and confidence is not None and confidence >= _MARKET_CONFIDENCE_MIN:
        return {"news_mult": _MARKET_MULT, "duration_h": _MARKET_DURATION_H, "scope": "market"}

    if scope == "coin" and severity == _SEVERE and category in _HACK_CATEGORIES:
        return {"target_zero": True, "block_h": _COIN_BLOCK_H, "coins": coins}

    return no_op


# TODO(Milestone 7, optional): live feedparser + anthropic wiring, see BUILD_SPEC.md Section 9.
# - Poll https://www.coindesk.com/arc/outboundfeeds/rss/, https://cointelegraph.com/rss,
#   https://www.theblock.co/rss.xml every 30 min with feedparser; dedupe by URL hash; only headlines
#   from the last 2 hours.
# - Call claude-haiku-4-5-20251001 via the anthropic SDK, temperature=0, max_tokens=400, key from
#   ANTHROPIC_API_KEY, with the Section 9 system prompt, to produce the `parsed` dict consumed above.
# - Log prompt, raw response, parsed JSON and the resulting action to logs/llm/{date}.jsonl.
# - On any API or parse failure: no-op (return {"news_mult": 1.0} equivalent upstream).
# This is I/O and therefore does NOT belong in this pure module -- it belongs in the engine/live layer,
# which should call apply_news_action() with the parsed result.
