# cfa-viral-lineage-model

⚠️ The work in this repository is in progress and highly experimental. ⚠️

## Overview

This repo hosts work on modeling how the composition of viral lineages, such as SARS-CoV-2 Pango lineages, changes over time.

The repo has the following structure:

- `linmod/`, with a package for downloading lineage count data, and fitting and evaluating models with this data; and
- `retrospective-forecasting/`, with code that uses this package to automate the retrospective evaluation of models.

### Architecture

The model is provided with lightly-preprocessed data of variant sequences from humans in the USA, from [Nextstrain](https://docs.nextstrain.org/projects/ncov/en/latest/reference/remote_inputs.html) ([data dictionary](https://docs.nextstrain.org/projects/ncov/en/latest/reference/metadata-fields.html)).
An Apache Parquet is provided, with columns `date`, `fd_offset`, `division`, `lineage`, `count`.
Rows are uniquely identified by `(date, division, lineage)`.
`date` and `fd_offset` can be computed from each other, given the forecast date.

Note that `date` is the sample collection date. `fd` refers to the forecast date. `fd_offset` is `date - fd` measured in days. Sequences are filtered to have a collection date no later than the forecast date.

| date       | fd_offset | division     | lineage | count |
| ---------- | ---------- | ------------ | ------- | ----- |
| 2024-05-07 | -12        | Arizona      | 24A     | 1     |
| 2024-05-04 | -15        | Pennsylvania | 24A     | 2     |
| ...        | ...        | ...          | ...     | ...   |

The model must output samples of population-level lineage proportions.
An Apache Parquet should be provided, with columns `fd_offset`, `division`, `lineage`, `sample_index`, and `phi` (the population proportion), for `fd_offset = -30, ..., 14`.
Rows are uniquely identified by `(fd_offset, division, lineage, sample_index)`.

| fd_offset | division | lineage | sample_index | phi            |
| ---------- | -------- | ------- | ------------ | -------------- |
| -30        | Alabama  | 22B     | 1            | 0.000014979599 |
| -30        | Alabama  | 22B     | 2            | 9.945703e-7    |
| ...        | ...      | ...     | ...          | ...            |


## Milestones & timeline

### Must-haves
- [x] (1) Implement baseline and starter models
	- Regression assuming spatial independence, no time covariate
	- Regression assuming spatial independence and a time covariate
- [x] (2) Design simulation study to verify model implementation
- [x] (3) Implement a couple metrics to evaluate population-level lineage proportion forecasts in a retrospective setting
- [x] (4) Conduct retrospective evaluations of our models with our metrics
- [x] (5) Prepare for symposium & friends

### Wishlist
- [ ] (6) Implement a metric to evaluate population-level lineage domination time predictions in a retrospective setting
	- Answer two questions: will lineage X take off? Given that lineage X takes off, at what time point does it reach 50% phi?
- [ ] (7) Can we obtain lineage growth rates in a model-agnostic way, from only posterior samples of population-level lineage proportions?
- [x] (8) Implement more advanced model and simulation study to verify
	- Regression with information sharing over space
- [ ] (9) Study more on how to set priors on the logit scale to induce priors on the probability simplex
- [ ] (10) Does our ability to identify "good" models change if we evaluate daily vs weekly predictions?

### A ladder of (multinomial logistic regression) models for consideration

It seems useful to start at the bottom of the ladder, both for debugging purposes, but also to get a sense of the added predictive power of each step.
You get more and more parameters to estimate; how does that balance against the improved accuracy?

1. One human population/geography (e.g. state, HHS region, country), two pathogen populations (dominant variant vs. everything else), binomial dynamics
   - i.e. `dominant% ~ invlogit[intercept + "slope" * time]`
3. One geography, multiple variants, multinomial
4. Multiple geographies, multiple variants, multinomial, no correlations or partial pooling
   - Statistically equivalent to #2, but requires some different computational implementation
5. As above plus partial pooling of "slopes" by variant across geographies, but without correlations:
   - `beta_1ij ~ N(mu_beta1i, sigma_beta1i)`
   - `mu_beta1i ~ some prior`, `sigma_beta1i ~ some prior`
   - `i=variant` `j=geography`
6. As above plus partial pooling of intercepts by variant across geographies
7. As above plus correlations between "slopes" by variant across counties
   - i.e. if growth rates of variant A and B are correlated across countries X and Y, then we also expect variants C and D to have correlated growth rates across countries X and Y
   - `beta_1*j ~ MVN(mu_beta1*, Sigma)`
   - `Sigma ~ some LKJ prior`

## General Disclaimer
This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).
