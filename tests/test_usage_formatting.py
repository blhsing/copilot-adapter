from copilot_adapter import _format_usage_details


def test_chatgpt_usage_uses_readable_window_labels_and_timing():
    details = {
        "rate_limit": {
            "primary_window": {
                "used_percent": 25,
                "used": 10,
                "limit": 40,
                "limit_window_seconds": 18000,
                "reset_after_seconds": 5400,
            },
            "secondary_window": {
                "used_percent": 50,
                "remaining": 5,
                "window_minutes": 10080,
                "resetAt": "2026-06-08T00:00:00Z",
            },
        },
    }

    assert _format_usage_details("chatgpt", details) == (
        "usage: 5-hour limit: 25% used, 10/40, resets in 1h 30m; "
        "7-day limit: 50% used, remaining 5, resets 2026-06-08 00:00 UTC"
    )


def test_chatgpt_usage_falls_back_to_common_window_lengths():
    details = {
        "rate_limit": {
            "primary_window": {"used_percent": 25},
            "secondary_window": {"used_percent": 50},
        },
    }

    assert _format_usage_details("chatgpt", details) == (
        "usage: 5-hour limit: 25% used; 7-day limit: 50% used"
    )


def test_anthropic_usage_uses_expanded_window_labels():
    details = {
        "five_hour": {
            "utilization": 12.5,
            "limit": 100,
            "remaining": 87.5,
        },
        "seven_day_opus": {
            "utilization": 80,
            "resets_at": "2026-06-08T05:30:00Z",
        },
    }

    assert _format_usage_details("anthropic", details) == (
        "usage: 5-hour limit: 12.5% used, limit 100, remaining 87.5; "
        "7-day Opus limit: 80% used, resets 2026-06-08 05:30 UTC"
    )
