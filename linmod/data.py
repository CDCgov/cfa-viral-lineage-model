"""
Download the Nextstrain metadata file, preprocess it, and export it.

Two datasets are exported: one for model fitting and one for evaluation.
The model dataset contains sequences collected and reported by a specified
forecast date, while the evaluation dataset extends the horizon into the future.

To change default behaviors, create a yaml configuration file with the key ["data"],
and pass it in the call to this script. For a list of configurable sub-keys, see the
`DEFAULT_CONFIG` dictionary.

The output is given in Apache Parquet format, with columns `date`, `fd_offset`,
`division`, `lineage`, `count`. Rows are uniquely identified by
`(date, division, lineage)`. `date` and `fd_offset` can be computed from each other,
given the forecast date; the `fd_offset` column is the number of days between the
forecast date and the `date` column, such that, for example, 0 is the forecast date,
-1 the day before, and 1 the day after.

Note that observations without a recorded date are removed, and only observations
from human hosts are included.
"""

import argparse
import gzip
import io
import lzma
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import cladecombiner
import polars as pl
import yaml
import zstandard

from .utils import ValidPath, expand_grid, print_message

DEFAULT_CONFIG = {
    "data": {
        # Where should the NextStrain data be downloaded from?
        "nextstrain_source": "https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst",
        # Should we use UShER to get the retrospective data?
        "use_usher": True,
        # We get UShER data from forecast_date + datetime.timedelta(days = usher_lag) rounded to the first of the month.
        # This is a compromise between recency of calls and maximizing available evaluation data, and accounts for
        # the archived data only being retained on the first of each month
        "usher_lag": 168,
        # Where should the UShER data be looked for? (Strong filepath assumptions are made about folders within this)
        "usher_root": "https://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/",
        # Should we use cladecombiner.AsOfAggregator to ensure lineages are only those known as of the forecast_date?
        "use_cladecombiner_as_of": False,
        # Where (directory) should the unprocessed (but decompressed) data be stored?
        "cache_dir": ".cache/",
        # Where (files) should the processed datasets for modeling and evaluation
        # be stored?
        "save_file": {
            "model": "data/metadata-model.parquet",
            "eval": "data/metadata-eval.parquet",
        },
        # Should the data be redownloaded (and the cache replaced)?
        "redownload": False,
        # What column should be renamed to `lineage` in the Nextstrain (reference) data?
        "nextstrain_lineage_column_name": "clade_nextstrain",
        # What column should be renamed to `lineage` in the UShER data?
        "usher_lineage_column_name": "Nextstrain_clade",
        # The as-of date that we wish to approximate running the pipeline on, may be the present
        # No sequences collected or reported after this date are included in the
        # modeling dataset.
        "forecast_date": {
            "year": datetime.now().year,
            "month": datetime.now().month,
            "day": datetime.now().day,
        },
        # How many days since the forecast date should be included in the datasets?
        # The evaluation dataset will contain sequences collected and reported within
        # this horizon. The modeling dataset will contain sequences collected and
        # reported within the horizon `[lower, 0]`.
        "horizon": {
            "lower": -90,
            "upper": 14,
        },
        # Which divisions should be included?
        # Currently set to the 50 U.S. states, D.C., and Puerto Rico
        "included_divisions": [
            "Alabama",
            "Alaska",
            "Arizona",
            "Arkansas",
            "California",
            "Colorado",
            "Connecticut",
            "Delaware",
            "Florida",
            "Georgia",
            "Hawaii",
            "Idaho",
            "Illinois",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "Minnesota",
            "Mississippi",
            "Missouri",
            "Montana",
            "Nebraska",
            "Nevada",
            "New Hampshire",
            "New Jersey",
            "New Mexico",
            "New York",
            "North Carolina",
            "North Dakota",
            "Ohio",
            "Oklahoma",
            "Oregon",
            "Pennsylvania",
            "Puerto Rico",
            "Rhode Island",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Vermont",
            "Virginia",
            "Washington",
            "Washington DC",
            "West Virginia",
            "Wisconsin",
            "Wyoming",
        ],
        # Which lineages should be included?
        # If not provided, all observed lineages are included.
        # If a list of length >= 1, all observed lineages not in this list are grouped
        # into "other".
        "lineages": [],
    }
}
"""
Default configuration for data download, preprocessing, and export.

The configuration dictionary expects all of the following entries in a
`data` key.
"""

hhs_regions = {
    "Connecticut": "HHS-1",
    "Maine": "HHS-1",
    "Massachusetts": "HHS-1",
    "New Hampshire": "HHS-1",
    "Rhode Island": "HHS-1",
    "Vermont": "HHS-1",
    "New Jersey": "HHS-2",
    "New York": "HHS-2",
    "Puerto Rico": "HHS-2",
    "Virgin Islands": "HHS-2",
    "Delaware": "HHS-3",
    "Washington DC": "HHS-3",
    "Maryland": "HHS-3",
    "Pennsylvania": "HHS-3",
    "Virginia": "HHS-3",
    "West Virginia": "HHS-3",
    "Alabama": "HHS-4",
    "Florida": "HHS-4",
    "Georgia": "HHS-4",
    "Kentucky": "HHS-4",
    "Mississippi": "HHS-4",
    "North Carolina": "HHS-4",
    "South Carolina": "HHS-4",
    "Tennessee": "HHS-4",
    "Illinois": "HHS-5",
    "Indiana": "HHS-5",
    "Michigan": "HHS-5",
    "Minnesota": "HHS-5",
    "Ohio": "HHS-5",
    "Wisconsin": "HHS-5",
    "Arkansas": "HHS-6",
    "Louisiana": "HHS-6",
    "New Mexico": "HHS-6",
    "Oklahoma": "HHS-6",
    "Texas": "HHS-6",
    "Iowa": "HHS-7",
    "Kansas": "HHS-7",
    "Missouri": "HHS-7",
    "Nebraska": "HHS-7",
    "Colorado": "HHS-8",
    "Montana": "HHS-8",
    "North Dakota": "HHS-8",
    "South Dakota": "HHS-8",
    "Utah": "HHS-8",
    "Wyoming": "HHS-8",
    "Arizona": "HHS-9",
    "California": "HHS-9",
    "Hawaii": "HHS-9",
    "Nevada": "HHS-9",
    "American Samoa": "HHS-9",
    "Commonwealth of the Northern Mariana Islands": "HHS-9",
    "Federated States of Micronesia": "HHS-9",
    "Guam": "HHS-9",
    "Marshall Islands": "HHS-9",
    "Republic of Palau": "HHS-9",
    "Alaska": "HHS-10",
    "Idaho": "HHS-10",
    "Oregon": "HHS-10",
    "Washington": "HHS-10",
}
"""
Dictionary form of https://www.hhs.gov/about/agencies/iea/regional-offices/index.html, except that DC is Washington DC not District of Columbia
"""


class CountsFrame(pl.DataFrame):
    """
    A `polars.DataFrame` which enforces a format for observed counts of lineages.

    See `REQUIRED_COLUMNS` for the expected columns.
    """

    REQUIRED_COLUMNS = {
        "date",
        "fd_offset",
        "division",
        "lineage",
        "count",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.validate()

    @classmethod
    def read_parquet(cls, *args, **kwargs):
        return cls(pl.read_parquet(*args, **kwargs))

    def validate(self):
        assert self.REQUIRED_COLUMNS.issubset(
            self.columns
        ), f"Missing required columns: ({', '.join(self.REQUIRED_COLUMNS - set(self.columns))})"

        assert (
            self.null_count().sum_horizontal().item() == 0
        ), "Null values detected in the dataset."

        assert self[
            "count"
        ].dtype.is_integer(), "Count column must be an integer type."


def process_nextstrain(
    fp: str,
    forecast_date,
    config: dict,
) -> pl.DataFrame:
    """
    Reads in Nextstrain data from (uncompressed) Nextstrain metadata file,
    performs basic filtering and date wrangling.
    """
    horizon_lower_date = forecast_date.dt.offset_by(
        f"{config['data']['horizon']['lower']}d"
    )
    horizon_upper_date = forecast_date.dt.offset_by(
        f"{config['data']['horizon']['upper']}d"
    )

    df = (
        pl.scan_csv(fp, separator="\t")
        .rename({config["data"]["nextstrain_lineage_column_name"]: "lineage"})
        # Cast with `strict=False` replaces invalid values with null,
        # which we can then filter out. Invalid values include dates
        # that are resolved only to the month, not the day
        .cast({"date": pl.Date, "date_submitted": pl.Date}, strict=False)
        .filter(
            # Drop samples with missing lineage
            pl.col("lineage").is_not_null(),
            # Drop samples with missing collection or reporting dates
            pl.col("date").is_not_null(),
            pl.col("date_submitted").is_not_null(),
            # Drop samples collected outside the horizon
            horizon_lower_date <= pl.col("date"),
            pl.col("date") <= horizon_upper_date,
            # Drop samples claiming to be reported before being collected
            pl.col("date") <= pl.col("date_submitted"),
            # Drop samples not from humans in the included US divisions
            pl.col("division").is_in(config["data"]["included_divisions"]),
            country="USA",
            host="Homo sapiens",
        )
        .collect()
    )
    return df


def recode_clades_using_usher(
    ns: pl.DataFrame,
    usher_path,
    usher_lineage_from: str,
    lineage_to="lineage",
) -> pl.DataFrame:
    """
    Replaces the "lineage" column in the input ns (Nextstrain) dataframe using
    clades called by UShER, as read from an UShER metadata file.

    Performs matching based on Genbank accessions. Unmatched entries are dropped.

    Useful for better-approximating retrospectively running an analysis
    as UShER metadata, including Nextstrain clade calls, are archived as far
    back as mid 2021.
    """
    usher = (
        pl.scan_csv(usher_path, separator="\t")
        .filter(pl.col("genbank_accession").is_not_null())
        .rename({"genbank_accession": "genbank_with_revision"})
        .unique("genbank_with_revision")
        .with_columns(
            pl.col(usher_lineage_from)
            # UShER names follow the Nextstrain_clade style not the clade_nextstrain style
            .str.extract(r"(\d\d[A-Z])").alias(lineage_to),
            # UShER includes the revision as part of the accession
            # we just want to match accessions
            genbank_accession=pl.col("genbank_with_revision").str.replace(
                r"\.(\d+)", ""
            ),
            genbank_revision=pl.col("genbank_with_revision")
            .str.extract(r"\.(\d+)")
            .cast(pl.Int64),
        )
        .with_columns(
            pl.when(pl.col(usher_lineage_from) == "recombinant")
            .then(pl.col(usher_lineage_from))
            .otherwise(pl.col(lineage_to))
            .alias(lineage_to),
        )
        # There are occasionally multiple revisions for the same accession,
        # take the most recent
        .filter(
            pl.col(lineage_to).is_not_null(),
            pl.col("genbank_revision")
            == pl.col("genbank_revision").max().over("genbank_accession"),
        )
        .select(["genbank_accession", "lineage"])
        .collect()
    )

    return ns.drop("lineage").join(
        usher, on="genbank_accession", how="inner", validate="1:1"
    )


def combine_clades(df: pl.DataFrame, as_of: date, lineage_col="lineage"):
    """
    Uses CDCGov/cladecombiner to recode the stated "lineage" such that
    any Nextstrain clade which was not recognized (had not yet been named)
    by the as-of date is put in its ancestor which was recognized.

    Useful for when taxonomy shifts (a new clade is named) within a few
    months of the desired `forecast_date` for better-approximating
    retrospectively running the pipeline.
    """
    ns = cladecombiner.nextstrain_sc2_nomenclature
    observed_clades = df[lineage_col].unique().to_list()

    assert len(observed_clades) > 0
    has_recombinants = "recombinant" in observed_clades
    if has_recombinants:
        observed_clades.remove("recombinant")

    if len(observed_clades) == 0:
        mapping = {"recombinant": "recombinant"}
    else:
        ns.validate(observed_clades)  # throws error if clades aren't valid
        taxa = [cladecombiner.Taxon(taxon, True) for taxon in observed_clades]
        tree = ns.taxonomy_tree(taxa)
        scheme = cladecombiner.PhylogeneticTaxonomyScheme(tree)
        aggregator = cladecombiner.AsOfAggregator(scheme, ns, as_of)
        mapping = aggregator.aggregate(taxa).to_str()
        if has_recombinants:
            mapping["recombinant"] = "recombinant"

    return df.with_columns(pl.col(lineage_col).replace_strict(mapping))


def main(cfg: Optional[dict]):
    config = DEFAULT_CONFIG

    if cfg is not None:
        config["data"] |= cfg["data"]

    # Download the data, if necessary

    parsed_url = urlparse(config["data"]["nextstrain_source"])
    nextstrain_cache_path = (
        ValidPath(config["data"]["cache_dir"])
        / parsed_url.netloc
        / parsed_url.path.lstrip("/").rsplit(".", 1)[0]
    )

    if config["data"]["redownload"] or not nextstrain_cache_path.exists():
        print_message("Downloading Nextstrain data...", end="")

        with (
            urlopen(config["data"]["nextstrain_source"]) as response,
            nextstrain_cache_path.open("wb") as out_file,
        ):
            if parsed_url.path.endswith(".xz"):
                with lzma.open(response) as in_file:
                    out_file.write(in_file.read())

            elif parsed_url.path.endswith(".zst"):
                decompressor = zstandard.ZstdDecompressor()

                with decompressor.stream_reader(response) as reader:
                    out_file.write(reader.readall())

            else:
                raise ValueError(f"Unsupported file format: {parsed_url.path}")

        print_message(" done.")
    else:
        print_message("Using cached Nextstrain data.")

    if config["data"]["use_usher"]:
        usher_date_unrounded = datetime(
            config["data"]["forecast_date"]["year"],
            config["data"]["forecast_date"]["month"],
            config["data"]["forecast_date"]["day"],
        ) + timedelta(days=config["data"]["usher_lag"])
        usher_date = datetime(
            usher_date_unrounded.year, usher_date_unrounded.month, 1
        )
        ymd = [
            str(usher_date.year),
            f"{usher_date.month:02d}",
            f"{usher_date.day:02d}",
        ]

        usher_url = (
            config["data"]["usher_root"]
            + "/".join(ymd)
            + "/public-"
            + "-".join(ymd)
            + ".metadata.tsv.gz"
        )

        usher_cache_path = (
            ValidPath(config["data"]["cache_dir"])
            / "usher"
            / ymd[0]
            / ymd[1]
            / ymd[2]
            / "metadata.tsv"
        )

        if config["data"]["redownload"] or not usher_cache_path.exists():
            print_message("Downloading UShER data...", end="")

            with (
                urlopen(usher_url) as response,
                usher_cache_path.open("wb") as out_file,
            ):
                compressed_file = response.read()
                f = gzip.GzipFile(fileobj=io.BytesIO(compressed_file))
                out_file.write(f.read())

            print_message(" done.")
        else:
            print_message("Using cached UShER data.")

    # Preprocess and export the data

    forecast_date = pl.date(
        config["data"]["forecast_date"]["year"],
        config["data"]["forecast_date"]["month"],
        config["data"]["forecast_date"]["day"],
    )

    full_df = process_nextstrain(nextstrain_cache_path, forecast_date, config)

    if config["data"]["use_usher"]:
        print_message("Using lineage assignments from UShER data.")
        full_df = recode_clades_using_usher(
            full_df,
            usher_path=usher_cache_path,
            usher_lineage_from=config["data"]["usher_lineage_column_name"],
        )

    if config["data"]["use_cladecombiner_as_of"]:
        print_message(
            "Ensuring only clades recognized as of the forecast date are present."
        )
        full_df = combine_clades(
            full_df,
            as_of=date(
                config["data"]["forecast_date"]["year"],
                config["data"]["forecast_date"]["month"],
                config["data"]["forecast_date"]["day"],
            ),
        )

    model_all_lineages = len(config["data"]["lineages"]) == 0
    full_df = full_df.with_columns(
        lineage=pl.when(
            pl.col("lineage").is_in(config["data"]["lineages"])
            | model_all_lineages
        )
        .then(pl.col("lineage"))
        .otherwise(pl.lit("other"))
    )

    print_message("Exporting evaluation dataset...", end="")
    # Generate every combination of date-division-lineage, so that:
    #  1. The evaluation dataset will be evaluation-ready, with 0 counts
    #     where applicable
    #  2. The modeling dataset will have every lineage of interest represented,
    #     even if a lineage was only sampled in the evaluation period
    observations_key = expand_grid(
        date=full_df["date"].unique(),
        division=full_df["division"].unique(),
        lineage=full_df["lineage"].unique(),
    )

    eval_df = CountsFrame(
        full_df.group_by("lineage", "date", "division")
        .agg(count=pl.len())
        .join(
            observations_key,
            on=("date", "division", "lineage"),
            how="right",
        )
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days(),
            count=pl.col("count").fill_null(0),
        )
        .select("date", "fd_offset", "division", "lineage", "count")
        # Sort to guarantee consistent output, since `.unique()` does not
        .sort("fd_offset", "division", "lineage")
    )

    eval_divisions = set(eval_df["division"].unique())
    if (
        not_included := set(config["data"]["included_divisions"]).difference(
            eval_divisions
        )
    ) != set():
        print_message(
            f" The following divisions have no data and have been dropped from the evaluation dataset: {not_included}"
        )

    eval_df.write_parquet(ValidPath(config["data"]["save_file"]["eval"]))

    print_message(" done.")
    print_message("Exporting modeling dataset...", end="")

    model_df = CountsFrame(
        full_df.filter(pl.col("date_submitted") <= forecast_date)
        .group_by("lineage", "date", "division")
        .agg(count=pl.len())
        .join(
            observations_key,
            on=("date", "division", "lineage"),
            how="right",
        )
        .with_columns(
            fd_offset=(pl.col("date") - forecast_date).dt.total_days(),
            count=pl.col("count").fill_null(0),
        )
        .select("date", "fd_offset", "division", "lineage", "count")
        # Remove division-days where no samples were collected, for brevity
        .filter(pl.sum("count").over("date", "division") > 0)
        # Sort to guarantee consistent output, since `.unique()` does not
        .sort("fd_offset", "division", "lineage")
    )

    assert set(model_df["lineage"].unique().to_list()) == set(
        eval_df["lineage"].unique().to_list()
    ), "Modeling and evaluation data have different lineages!"

    missing_model_divisions = eval_divisions.difference(
        model_df["division"].unique()
    )
    if missing_model_divisions:
        print_message(
            f" The following divisions have evaluation data but no modeling data: {missing_model_divisions}"
        )

    # Ensure every division is present on the forecast date, with 0 counts where no data is available
    data_on_0 = (
        model_df.filter(pl.col("fd_offset") == 0)["division"]
        .unique()
        .to_list()
    )
    pad_divisions = list(eval_divisions.difference(data_on_0))
    lineages = model_df["lineage"].unique().to_list()
    zero_df = (
        pl.DataFrame({"lineage": lineages})
        .with_columns(
            count=pl.lit(0),
            date=pl.lit(
                date(
                    config["data"]["forecast_date"]["year"],
                    config["data"]["forecast_date"]["month"],
                    config["data"]["forecast_date"]["day"],
                )
            ),
            fd_offset=pl.lit(0),
        )
        .cast({"count": pl.UInt32, "fd_offset": pl.Int64})
        .join(
            pl.DataFrame({"division": pad_divisions}),
            how="cross",
        )
        .select("date", "fd_offset", "division", "lineage", "count")
    )
    model_df = pl.concat([model_df, zero_df])

    model_divisions = set(model_df["division"].unique())
    assert (
        model_divisions == eval_divisions
    ), "Evaluation and modeling data contain different divisions!"

    assert model_df.filter(
        pl.col("fd_offset") == 0,
        pl.col("lineage") == model_df["lineage"].to_list()[0],
    ).shape[0] == len(eval_divisions), "Modeling data is incorrectly padded."

    all_data = model_df.join(
        eval_df,
        on=["date", "fd_offset", "division", "lineage"],
        how="full",
        validate="1:1",
    ).with_columns(eval_surplus=(pl.col("count_right") - pl.col("count")))
    assert (
        all_data["eval_surplus"] >= 0
    ).all(), "Evaluation dataset must have at least as many sequences as modeling data"

    model_df.write_parquet(ValidPath(config["data"]["save_file"]["model"]))

    print_message(" done.")

    model_all_lineages = len(config["data"]["lineages"]) == 0
    if model_all_lineages:
        print_message(
            "Modeling all lineages observed in the data at any point in the horizon."
        )
    else:
        print_message(
            (
                "Modeling the following subset of lineages, "
                '(all other lineages grouped into "other"): '
            )
            + str(config["data"]["lineages"])
        )


if __name__ == "__main__":
    # Load configuration, if given

    parser = argparse.ArgumentParser(
        prog="python3 -m linmod.data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    yaml_path = parser.parse_args().config

    cfg = None
    if yaml_path is not None:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

    main(cfg)
