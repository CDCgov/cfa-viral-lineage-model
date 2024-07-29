# Terminology and notation

Viral lineage models are high-dimensional, and there are many nuances involved. This document exists as a basis for standardizing how we will write descriptions and equations of these models to facilitate communication.

## Terminology

### Lineages

We will use _lineage_ as a catch-all term for the modeling targets, whether these are Pango lineages, NextStrain clades, or (potentially non-monophyletic) aggregations thereof.

Lineages are _assigned_ to sequences by some program (primarily NextClade and Pango for SARS-CoV-2 sequences). We take the observation of an assigned lineage to be data, but in truth it is not, as the result of lineage assignment depends on the version of the software used (and where applicable the reference tree).

### Dates

There are three dates pertinent to any observation in a dataset.

- The _event_ date is the day on which something happened. For example, the date on which a sample was collected.
- The _report_ date is the day on which the event was reported. For example, the day a sequence was submitted based on a collected sample.
- The _vintage_ date is the day on which the data was pulled from wherever it was hosted. For example, the day one downloaded the current NextStrain open data.

The vintage date is in a sense a proxy for the often unreported version details of software used to assign lineages to sequences. The sequence must have been assigned a lineage by a version no more recent than the one available on the vintage date.

## Notation

### Preliminaries

We will reserve $\theta$ and $\boldsymbol{\theta}$ for "some arbitrary parameter(s)." Semantic subscripts are allowed, for "some arbitrary parameters relating to [some part of the model]."

Vectors and matrices are denoted via `\mathbf`.

- $\mathbf{x}$ is a vector and element $i$ is $x_i$.
- $\mathbf{X}$ is a matrix, with element $ij$ being $X_{ij}$, and it can have row or column vectors $\mathbf{X}_{i \cdot}$ or $\mathbf{X}_{\cdot j}$ as needed.

We index semantically where possible.

- We reserve $t$ for time, $\ell$ for lineages, $g$ for the groups being modeled (e.g. hospitalized infections in Iowa), and when needed $s$ for data streams/sources/signals.
- We leave $i$, $j$, and $k$ to be available as-needed.

By aggregating samples in time, we are implicitly working in discrete time. We will therefor use $x_t$ for "the data/parameter value/etc at time $t$."

### Lineages

The latent, unobservable _true_ frequency of lineage $\ell$ in the population is in reality a discrete frequency. It is the number of people infected by lineage $\ell$ divided by the number of people infected by any lineage. We:

- Refer to it as the _population frequency_ of the lineage.
- Model it as if it were continuous.
- Absent a notion of space or time, denote it $\phi$ (for _frequency_).
- Spatiotemporal slices of all space x time x lineage combinations allow us to refer to $\boldsymbol{\phi}$ and $\boldsymbol{\Phi}$. We use
    - $\boldsymbol{\phi}_{t g}$ for all _lineages_ at some time in some group.
    - $\boldsymbol{\Phi}_t$ for all _lineages_ in all _groups_ at some time
    - $\boldsymbol{\Phi}$ for all lineages in all groups at all times.
    - The order of precedence of subscripts is lineage $l$, time $t$, and group $g$. For one single element, we have $\phi_{t g \ell}$.

### Data and covariates

A single _group_ $g$ corresponds to some population in which there is a single (unknown) population frequency of the lineages at any given time, $\boldsymbol{\phi}_{t g}$. This can be measured by more than one data source/stream.

#### Data

Our data in aggregate is $\mathbf{Y}$. This consists of potentially multiple data streams, which may capture disjoint or overlapping groups. When there are multiple data streams, $\mathbf{Y}_{s}$ indexes the data in one of them. Depending on the data stream, the observations $y_{s t g \ell}$ may vary. Primarily they will be _counts_ of lineages (in that data stream at some time in some group of some lineage).

#### Data dates

Data vintages can be referred to using $\mathbf{Y}(\tau)$. This implies all samples with report times $t < \tau$.

#### Sample statistics of data

Where two or more data streams both contain the same group $g$, there will be multiple _sample frequencies_ of any particular lineage $\ell$. We denote sample frequencies with $f$, such that $\mathbf{f}_{s t g}$ is the vector of all lineage sample frequencies in data stream $s$ at time $t$ in group $g$ and $\mathbf{F}_{s t}$ is the matrix of sample frequencies of all lineages in all regions in stream $s$ at time $t$. If we have count data, then $f_{s t g \ell} = y_{s t g \ell} / \sum_k y_{s t g k}$.

#### Covariates

Should we need covariates in the model, we will denote them $\mathbf{X}$.

### Model and model components

_The_ model consists of two components:

- A _lineage dynamic model_, which specifies $\text{Pr}(\boldsymbol{\Phi})$
- An _observation model_ which specifies $\text{Pr}(\mathbf{Y} \mid \boldsymbol{\Phi})$

These components may be highly complex ($\text{Pr}(\boldsymbol{\Phi})$ could be a spatiotemporal Gaussian process) or very simple ($\text{Pr}(\mathbf{Y}_{s t g} \mid \boldsymbol{\Phi})$ could simply be that the observations are independent draws from a Multinomial($n$, $\boldsymbol{\phi}_{t g}$) for all times $t$).
