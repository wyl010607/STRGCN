import pandas as pd


def parse_duration_to_seconds(expr) -> float:
    """
    Parse a duration expression into seconds (can be fractional).

    Parameters
    ----------
    expr : str | int | float
        Examples: "1h", "15min", "4s", "250ms", 3600.

    Returns
    -------
    float
        Duration in seconds. May contain fractional part (e.g., 0.25 for 250ms).
    """
    if isinstance(expr, (int, float)):
        if expr < 0:
            raise ValueError("Duration cannot be negative.")
        return float(expr)

    expr = str(expr).strip()
    try:
        return pd.to_timedelta(expr).total_seconds()
    except ValueError:
        raise ValueError(f"Unrecognised duration: {expr!r}")
