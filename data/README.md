Usage: `load_metadata.py [URL]`

Download the Nextstrain metadata file from the given `URL`, preprocess it,
and print the result to `stdout`.

The output is given in CSV format, with columns `lineage`, `date`, `division`,
and `count`. Rows are uniquely identified by `(lineage, date, division)`.

Preprocessing is done to ensure that:
- Observations without a recorded date are removed;
- Only the 50 U.S. states, D.C., and Puerto Rico are included; and
- Only observations from human hosts are included.

If no `URL` is given, defaults to:
https://data.nextstrain.org/files/ncov/open/metadata.tsv.zst
