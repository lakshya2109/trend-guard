from scripts.competition_snapshot import regional_rows, selected_rows, summarize_portfolio


def test_regional_rows_sort_by_profit_then_overall_rank():
    board = {
        "PublicRank": [
            {"CountryCode": "AU", "Profit": 0.2, "Rank": 1, "UserCode": "AU01"},
            {"CountryCode": "HK", "Profit": -0.01, "Rank": 8, "UserCode": "HK02"},
            {"CountryCode": "HK", "Profit": 0.03, "Rank": 4, "UserCode": "HK01"},
            {"CountryCode": "HK", "Profit": -0.01, "Rank": 7, "UserCode": "HK03"},
        ]
    }

    assert [row["UserCode"] for row in regional_rows(board, "HK")] == ["HK01", "HK03", "HK02"]


def test_selected_rows_include_top_cutoff_and_own_without_duplicates():
    rows = [{"UserCode": f"HK{i:02d}"} for i in range(1, 26)]

    chosen = selected_rows(rows, own_code="HK23", top=3, cutoff_rank=20)

    assert [(rank, row["UserCode"]) for rank, row in chosen] == [
        (1, "HK01"),
        (2, "HK02"),
        (3, "HK03"),
        (20, "HK20"),
        (23, "HK23"),
    ]


def test_summarize_portfolio_counts_material_longs_and_open_shorts():
    summary = summarize_portfolio(
        {
            "DelayedMinute": 5,
            "USDBalance": 20_000,
            "CoinProfit": [
                {"CoinLeft": 2, "EstimatedPrice": 10_000},
                {"CoinLeft": 0.1, "EstimatedPrice": 5},
                {"CoinLeft": 0, "EstimatedPrice": 100},
            ],
            "ShortPositions": [
                {"PositionStatus": "OPEN", "PositionValue": 15_000},
                {"PositionStatus": "CLOSED", "PositionValue": 3_000},
            ],
        }
    )

    assert summary.cash == 20_000
    assert summary.long_value == 20_000
    assert summary.short_value == 15_000
    assert summary.equity == 55_000
    assert summary.gross == 35_000
    assert summary.net == 5_000
    assert summary.long_count == 1
    assert summary.short_count == 1
    assert summary.delayed_minutes == 5
