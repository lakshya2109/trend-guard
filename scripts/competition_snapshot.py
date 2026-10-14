"""Read-only HK leaderboard and delayed portfolio snapshot from Roostoo's public web API.

    python scripts/competition_snapshot.py
    python scripts/competition_snapshot.py --region HK --top 10 --cutoff-rank 20

This never uses competition credentials and cannot place orders. Portfolio views are delayed and
show current posture only, not another team's strategy or historical exposure.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests

BASE_URL = "https://mock-api.roostoo.com"
APP_VERSION = "57"


@dataclass(frozen=True)
class PortfolioSummary:
    cash: float
    long_value: float
    short_value: float
    long_count: int
    short_count: int
    delayed_minutes: int

    @property
    def equity(self) -> float:
        return self.cash + self.long_value + self.short_value

    @property
    def gross(self) -> float:
        return self.long_value + self.short_value

    @property
    def net(self) -> float:
        return self.long_value - self.short_value


def _get(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = {"timestamp": int(time.time() * 1000), **params}
    response = session.get(
        BASE_URL + path,
        params=query,
        headers={"VERSION": APP_VERSION, "APP_LANGUAGE": "en"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("Success", False):
        raise RuntimeError(data.get("ErrMsg") or f"{path} failed")
    return data


def regional_rows(data: dict[str, Any], region: str) -> list[dict[str, Any]]:
    rows = [row for row in data.get("PublicRank", []) if row.get("CountryCode") == region]
    return sorted(rows, key=lambda row: (-float(row["Profit"]), int(row["Rank"])))


def summarize_portfolio(data: dict[str, Any]) -> PortfolioSummary:
    longs = [
        float(row.get("CoinLeft") or 0) * float(row.get("EstimatedPrice") or 0)
        for row in data.get("CoinProfit", [])
    ]
    open_shorts = [
        row for row in data.get("ShortPositions", []) if row.get("PositionStatus") == "OPEN"
    ]
    return PortfolioSummary(
        cash=float(data.get("USDBalance") or 0),
        long_value=sum(value for value in longs if value >= 1),
        short_value=sum(float(row.get("PositionValue") or 0) for row in open_shorts),
        long_count=sum(value >= 1 for value in longs),
        short_count=len(open_shorts),
        delayed_minutes=int(data.get("DelayedMinute") or 0),
    )


def selected_rows(
    rows: list[dict[str, Any]], own_code: str, top: int, cutoff_rank: int
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for regional_rank, row in enumerate(rows, 1):
        if regional_rank <= top or regional_rank == cutoff_rank or row.get("UserCode") == own_code:
            selected.append((regional_rank, row))
    return selected


def _money(value: float) -> str:
    return f"${value:,.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-id", type=int, default=538)
    parser.add_argument("--region", default="HK")
    parser.add_argument("--own-code", default="CGJY")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--cutoff-rank", type=int, default=20)
    parser.add_argument(
        "--leaderboard-only",
        action="store_true",
        help="skip delayed portfolio calls and print values/returns only",
    )
    args = parser.parse_args(argv)

    with requests.Session() as session:
        board = _get(
            session,
            "/v1/leader_board",
            {"competition_id": args.competition_id, "lb_level": "OVERALL"},
        )
        rows = regional_rows(board, args.region.upper())
        chosen = selected_rows(rows, args.own_code, args.top, args.cutoff_rank)
        if not chosen:
            raise RuntimeError(f"no {args.region.upper()} leaderboard rows returned")

        updated = datetime.fromtimestamp(float(board["LastUpdate"]) / 1000, UTC)
        print(
            f"Roostoo competition {args.competition_id} · {args.region.upper()} · "
            f"leaderboard updated {updated:%Y-%m-%d %H:%M:%S UTC}"
        )
        print(
            "rank  overall  code  equity        return   volume        "
            "cash    gross    net      positions  team"
        )

        own: tuple[int, dict[str, Any]] | None = None
        cutoff: tuple[int, dict[str, Any]] | None = None
        delay = 0
        for regional_rank, row in chosen:
            if row.get("UserCode") == args.own_code:
                own = (regional_rank, row)
            if regional_rank == args.cutoff_rank:
                cutoff = (regional_rank, row)

            posture = ""
            if not args.leaderboard_only:
                portfolio = _get(
                    session,
                    "/v6/portfolio_visitor",
                    {
                        "competition_id": args.competition_id,
                        "user_code": row["UserCode"],
                    },
                )
                summary = summarize_portfolio(portfolio)
                delay = max(delay, summary.delayed_minutes)
                equity = max(summary.equity, 1.0)
                posture = (
                    f"{summary.cash / equity:6.1%}  {summary.gross / equity:6.1%}  "
                    f"{summary.net / equity:+6.1%}  "
                    f"{summary.long_count:2d}L/{summary.short_count:2d}S"
                )

            print(
                f"{regional_rank:>4}  {int(row['Rank']):>7}  {row['UserCode']:<4}  "
                f"{_money(float(row['CurrBal'])):>12}  {float(row['Profit']):+7.2%}  "
                f"{_money(float(row['TradeVolume'])):>12}  {posture:<31}  "
                f"{row.get('DisplayName', '')}"
            )

        if own and cutoff:
            gap = float(cutoff[1]["CurrBal"]) - float(own[1]["CurrBal"])
            gap_return = gap / float(own[1]["CurrBal"])
            print(
                f"\n{args.own_code} is {args.region.upper()} #{own[0]}; "
                f"static gap to #{args.cutoff_rank}: {_money(gap)} ({gap_return:+.2%})."
            )
        if delay:
            print(
                f"Portfolio postures are delayed by up to {delay} minutes and show current "
                "holdings only."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
