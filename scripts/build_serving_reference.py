from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import load_tabular_data
from src.preprocessing import clean_energy_dataset
from src.settings import DATE_COL, TARGET_COL


DATA_PATH = Path("data/processed/damavand.csv")
OUTPUT_PATH = Path(
    "config/serving/input_reference.json"
)

DEVELOPMENT_START = pd.Timestamp("2025-09-12")
DEVELOPMENT_END = pd.Timestamp("2025-10-24")

RAW_NUMERIC_FEATURES = [
    "total_kg",
    "total_nominal_kg",
    "total_brix_units",
    "total_hours",
    "total_pallets",
    "orders",
    "avg_brix",
]

REFERENCE_NUMERIC_FEATURES = [
    *RAW_NUMERIC_FEATURES,
    "yield_ratio_actual_over_nominal",
]

CALENDAR_FEATURES = [
    "weekday",
    "is_weekend",
    "month",
    "year",
    "week_of_year",
]


def _numeric_summary(
    series: pd.Series,
) -> dict[str, float]:
    """Calculate serving boundaries for one numeric feature."""

    numeric = pd.to_numeric(series, errors="raise")

    return {
        "minimum": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "p90": float(numeric.quantile(0.90)),
        "maximum": float(numeric.max()),
    }


def main() -> None:
    data = load_tabular_data(DATA_PATH)
    data = clean_energy_dataset(
        data,
        DATE_COL,
        TARGET_COL,
    )

    data[DATE_COL] = pd.to_datetime(
        data[DATE_COL],
        dayfirst=True,
        errors="raise",
    )

    development = data.loc[
        data[DATE_COL].between(
            DEVELOPMENT_START,
            DEVELOPMENT_END,
            inclusive="both",
        )
    ].copy()

    development = development.sort_values(
        DATE_COL
    ).reset_index(drop=True)

    if len(development) != 37:
        raise ValueError(
            "Expected 37 post-only development rows "
            f"(30 training + 7 validation), found "
            f"{len(development)}."
        )

    development[
        "yield_ratio_actual_over_nominal"
    ] = (
        development["total_kg"]
        / development["total_nominal_kg"]
    )

    development["weekday"] = (
        development[DATE_COL].dt.weekday
    )
    development["is_weekend"] = (
        development["weekday"].isin([5, 6]).astype(int)
    )
    development["month"] = (
        development[DATE_COL].dt.month
    )
    development["year"] = (
        development[DATE_COL].dt.year
    )
    development["week_of_year"] = (
        development[DATE_COL]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    numeric_reference = {
        feature: _numeric_summary(
            development[feature]
        )
        for feature in REFERENCE_NUMERIC_FEATURES
    }

    calendar_reference = {
        feature: sorted(
            int(value)
            for value in development[feature].unique()
        )
        for feature in CALENDAR_FEATURES
    }

    reference = {
        "reference_version": "1.0",
        "reference_scope": (
            "post_only_development_train_plus_validation"
        ),
        "development_start": (
            DEVELOPMENT_START.date().isoformat()
        ),
        "development_end": (
            DEVELOPMENT_END.date().isoformat()
        ),
        "development_rows": len(development),
        "typical_lower_quantile": 0.10,
        "typical_upper_quantile": 0.90,
        "numeric_features": numeric_reference,
        "calendar_features": calendar_reference,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            reference,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT_PATH}")
    print(
        "Reference period:",
        reference["development_start"],
        "to",
        reference["development_end"],
    )
    print(
        "Development rows:",
        reference["development_rows"],
    )
    print(
        "Numeric features:",
        len(reference["numeric_features"]),
    )
    print(
        "Calendar features:",
        len(reference["calendar_features"]),
    )


if __name__ == "__main__":
    main()