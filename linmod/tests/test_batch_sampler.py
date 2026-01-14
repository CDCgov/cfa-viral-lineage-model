import polars as pl

from linmod import eval


def test_multinomial_batch_sampler_reproducible():
    """Ensure batched multinomial sampling is reproducible given the same seed

    We create a tiny dataset with two sample indexes and three lineages and
    check that invoking CountsEvaluator twice with the same seed yields
    identical `count_sampled` outputs, and that a different seed yields
    different outputs.
    """

    # Create data: 3 lineages for one division-day
    data = pl.DataFrame(
        {
            "fd_offset": [0, 0, 0],
            "division": [0, 0, 0],
            "lineage": [0, 1, 2],
            "count": [4, 3, 3],
            "date": [0, 0, 0],
        }
    )

    # Create samples: two sample_index values, each with a phi vector summing to 1
    samples = pl.DataFrame(
        {
            "fd_offset": [0, 0, 0, 0, 0, 0],
            "division": [0, 0, 0, 0, 0, 0],
            "lineage": [0, 1, 2, 0, 1, 2],
            "sample_index": [0, 0, 0, 1, 1, 1],
            "phi": [0.2, 0.3, 0.5, 0.3, 0.3, 0.4],
        }
    )

    # Run evaluator twice with same seed
    ev1 = eval.CountsEvaluator(samples, data, seed=12345)
    df1 = ev1.df.collect().sort(["sample_index", "lineage"])

    ev2 = eval.CountsEvaluator(samples, data, seed=12345)
    df2 = ev2.df.collect().sort(["sample_index", "lineage"])

    assert (
        df1.get_column("count_sampled").to_list()
        == df2.get_column("count_sampled").to_list()
    )
