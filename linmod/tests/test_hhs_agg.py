import polars as pl
import pytest
from polars.testing import assert_frame_equal

from linmod.models import InfectionWeightedAggregator


def test_infection_weighted_aggregator():
    one_state_level_sample = pl.DataFrame(
        {
            "fd_offset": [*([-10] * 12), *([12] * 12)],
            "division": [
                *(["Arizona"] * 3),
                *(["California"] * 3),
                *(["Idaho"] * 3),
                *(["Washington"] * 3),
            ]
            * 2,
            "lineage": [
                "99A",
                "99B",
                "99C",
            ]
            * 8,
            "phi": [
                0.2,
                0.3,
                0.5,
                0.3,
                0.3,
                0.4,
                0.2,
                0.3,
                0.5,
                0.3,
                0.3,
                0.4,
                0.1,
                0.3,
                0.6,
                0.2,
                0.3,
                0.5,
                0.1,
                0.3,
                0.6,
                0.2,
                0.3,
                0.5,
            ],
        }
    )

    state_level = pl.concat(
        [
            one_state_level_sample.with_columns(sample_index=pl.lit(25.0)),
            one_state_level_sample.with_columns(sample_index=pl.lit(42.0)),
        ]
    )

    pop_size = pl.DataFrame(
        {
            "division": ["Arizona", "California", "Idaho", "Washington"],
            "pop_size": [8, 40, 2, 8],
        }
    )

    prop_infected = pl.DataFrame(
        {
            "division": ["Arizona", "California", "Idaho", "Washington"],
            "prop_infected": [0.25, 0.5, 0.75, 0.25],
        }
    )

    full_geo_map = {
        "Arizona": "HHS-9",
        "California": "HHS-9",
        "Idaho": "HHS-10",
        "Washington": "HHS-10",
    }

    partial_geo_map = {
        "Arizona": "HHS-9",
        "California": "HHS-9",
        "Idaho": "HHS-10",
    }

    extra_geo_map = {
        "Arizona": "HHS-9",
        "California": "HHS-9",
        "Nevada": "HHS-9",
        "Idaho": "HHS-10",
        "Washington": "HHS-10",
    }

    expected_full_eq_wt = pl.DataFrame(
        {
            "sample_index": [42.0] * 12,
            "fd_offset": [*([-10] * 6), *([12] * 6)],
            "division": [
                *(["HHS-9"] * 3),
                *(["HHS-10"] * 3),
            ]
            * 2,
            "lineage": [
                "99A",
                "99B",
                "99C",
            ]
            * 4,
            "phi": [
                0.25,
                0.3,
                0.45,
                0.25,
                0.3,
                0.45,
                0.15,
                0.3,
                0.55,
                0.15,
                0.3,
                0.55,
            ],
        }
    )

    assert_frame_equal(
        (
            InfectionWeightedAggregator()(state_level, full_geo_map).filter(
                pl.col("sample_index") == 42.0
            )
        ),
        expected_full_eq_wt,
        check_row_order=False,
        check_column_order=True,
    )

    expected_full_no_prop = pl.DataFrame(
        {
            "sample_index": [42.0] * 12,
            "fd_offset": [*([-10] * 6), *([12] * 6)],
            "division": [
                *(["HHS-9"] * 3),
                *(["HHS-10"] * 3),
            ]
            * 2,
            "lineage": [
                "99A",
                "99B",
                "99C",
            ]
            * 4,
            "phi": [
                0.28333333,
                0.3,
                0.41666667,
                0.28,
                0.3,
                0.42,
                0.18333333,
                0.3,
                0.51666667,
                0.18,
                0.3,
                0.52,
            ],
        }
    )

    assert_frame_equal(
        (
            InfectionWeightedAggregator()(
                state_level, full_geo_map, pop_size=pop_size
            ).filter(pl.col("sample_index") == 42.0)
        ),
        expected_full_no_prop,
        check_row_order=False,
        check_column_order=True,
    )

    expected_full = pl.DataFrame(
        {
            "sample_index": [42.0] * 12,
            "fd_offset": [*([-10] * 6), *([12] * 6)],
            "division": [
                *(["HHS-9"] * 3),
                *(["HHS-10"] * 3),
            ]
            * 2,
            "lineage": [
                "99A",
                "99B",
                "99C",
            ]
            * 4,
            "phi": [
                0.29090909,
                0.3,
                0.40909091,
                0.25714286,
                0.3,
                0.44285714,
                0.19090909,
                0.3,
                0.50909091,
                0.15714286,
                0.3,
                0.54285714,
            ],
        }
    )

    assert_frame_equal(
        (
            InfectionWeightedAggregator()(
                state_level,
                full_geo_map,
                pop_size=pop_size,
                prop_infected=prop_infected,
            ).filter(pl.col("sample_index") == 42.0)
        ),
        expected_full,
        check_row_order=False,
        check_column_order=True,
    )

    expected_partial_no_prop = pl.DataFrame(
        {
            "sample_index": [42.0] * 12,
            "fd_offset": [*([-10] * 6), *([12] * 6)],
            "division": [
                *(["HHS-9"] * 3),
                *(["HHS-10"] * 3),
            ]
            * 2,
            "lineage": [
                "99A",
                "99B",
                "99C",
            ]
            * 4,
            "phi": [
                0.28333333,
                0.3,
                0.41666667,
                0.2,
                0.3,
                0.5,
                0.18333333,
                0.3,
                0.51666667,
                0.1,
                0.3,
                0.6,
            ],
        }
    )

    assert_frame_equal(
        (
            InfectionWeightedAggregator()(
                state_level, partial_geo_map, pop_size=pop_size
            ).filter(pl.col("sample_index") == 42.0)
        ),
        expected_partial_no_prop,
        check_row_order=False,
        check_column_order=True,
    )

    with pytest.raises(AssertionError):
        InfectionWeightedAggregator()(
            state_level, extra_geo_map, pop_size=pop_size
        )

    with pytest.raises(AssertionError):
        InfectionWeightedAggregator()(
            state_level,
            full_geo_map,
            pop_size=prop_infected.filter(
                pl.col("division").is_in(["Arizona", "Idaho", "Washington"])
            ),
        )

    with pytest.raises(AssertionError):
        InfectionWeightedAggregator()(
            state_level,
            full_geo_map,
            pop_size=prop_infected.filter(
                pl.col("division").is_in(["Arizona", "Idaho", "Washington"])
            ),
        )
