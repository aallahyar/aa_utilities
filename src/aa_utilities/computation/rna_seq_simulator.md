# Bulk RNA-seq Simulation Framework: Planning Specification

## Executive summary

This project will produce a **reproducible bulk RNA-seq count simulator** for three related purposes:

- **Differential-expression benchmarking**
- **Statistical power analysis**
- **Testing confounder adjustment**

The simulator will generate three linked pandas-compatible datasets:

1. A **gene-by-sample integer count matrix**
2. **Gene metadata** containing the simulation truth
3. **Sample metadata** containing conditions, confounders, and sample-level parameters

The core model will use **negative-binomial count generation**, gene-specific baseline expression, gene-specific dispersion, sample-specific library-size factors, condition effects, and confounder effects.

The first version will support arbitrary named conditions with unequal sample sizes, two categorical confounders, and one continuous confounder.

---

# 1. Project objectives

The simulator should support controlled experiments in which users know the true underlying parameters. This will allow them to evaluate whether downstream methods correctly recover:

- Differentially expressed genes
- True effect sizes
- Statistical significance
- False-positive rates
- Statistical power
- Effects of confounding
- The benefit or failure of covariate adjustment

The simulator should prioritize **interpretability, reproducibility, and controlled truth** over maximum biological realism.

It should be possible to deliberately generate:

- Balanced designs
- Partially confounded designs
- Strongly confounded designs
- Null simulations
- Simulations with known differential-expression signals
- Simulations with different sample sizes and effect sizes

---

# 2. Experimental condition specification

Conditions will be provided as a mapping from condition names to sample sizes.

Example:

```python
{
    "control": 6,
    "case_1": 6,
    "case_2": 6
}
```

The simulator should support:

- Any number of conditions
- Unequal sample sizes
- User-defined condition names
- A user-defined reference condition
- Automatic generation of sample IDs

The recommended default reference condition is `"control"` when it is present. Otherwise, the user should specify the reference condition explicitly.

For a multi-condition design, condition effects will be represented relative to the reference condition:

- `case_1` versus `control`
- `case_2` versus `control`
- `case_3` versus `control`

Pairwise contrasts can be calculated later from these reference-based effects.

---

# 3. Confounder support

The first version will support three user-configurable confounders:

- **Two categorical confounders**, such as batch and gender
- **One continuous confounder**, such as age

Example sample-level variables:

```text
batch: batch_1, batch_2
gender: female, male
age: continuous numeric values
```

## 3.1 Confounder assignment

The simulator should support several assignment strategies:

- **Balanced**: confounder levels are approximately equally distributed across conditions
- **Random**: values are assigned independently of condition
- **Partially confounded**: confounder distributions differ across conditions
- **Strongly confounded**: one or more confounder levels are concentrated in particular conditions
- **User-defined**: the user supplies the complete sample metadata or confounder values

A confounder is not defined only by having an effect on expression. It is a confounder when it is also associated with the biological condition being tested.

The simulator should therefore report condition-by-confounder summaries, such as:

- Condition-by-batch contingency tables
- Condition-by-gender contingency tables
- Mean and standard deviation of age by condition
- A summary of the degree of association between each condition and confounder

## 3.2 Confounder effects

Confounder effects will be specified on the **log2 fold-change scale** for categorical variables and on a defined per-unit scale for continuous variables.

For a categorical confounder, the simulator should use a reference level. For example:

```text
batch_2 versus batch_1
male versus female
```

For a continuous confounder such as age, the recommended approach is to center and scale the variable before applying its effect. This makes the effect size easier to interpret and avoids numerical problems caused by arbitrary measurement units.

A continuous effect can then be interpreted as the change in expression associated with a one-standard-deviation increase in age.

---

# 4. Count-generation model

Counts will be generated using a negative-binomial model.

For gene \(g\) and sample \(i\), the count will be represented as:

```math
Y_{gi} \sim \operatorname{NegativeBinomial}(\mu_{gi}, \phi_g)
```

The expected count will depend on baseline expression, library size, condition effects, and confounder effects.

A recommended log-scale model is:

```math
\log(\mu_{gi}) =
\log(s_i)
+
\alpha_g
+
\beta_{g,c(i)}
+
\gamma_{g,1}X_{i,1}
+
\gamma_{g,2}X_{i,2}
+
\gamma_{g,3}Z_i
```

where:

- \(Y_{gi}\) is the simulated count
- \(\mu_{gi}\) is the expected count
- \(s_i\) is the sample-specific library-size factor
- \(\alpha_g\) is the baseline expression for gene \(g\)
- \(\beta_{g,c(i)}\) is the condition effect
- \(X_{i,1}\) and \(X_{i,2}\) are categorical confounder indicators
- \(Z_i\) is the standardized continuous confounder
- \(\gamma\) values are confounder effects
- \(\phi_g\) is the gene-specific dispersion

The simulator should document the exact negative-binomial parameterization because different packages use different definitions of dispersion.

A recommended variance relationship is:

```math
\operatorname{Var}(Y_{gi}) =
\mu_{gi} + \phi_g\mu_{gi}^{2}
```

This is compatible with the common interpretation of a gene-specific negative-binomial dispersion.

---

# 5. Baseline expression

Each gene should have its own baseline expression level.

The initial simulator should generate baseline expression from a configurable parametric distribution, preferably a log-normal or gamma distribution. This provides heterogeneity across genes without requiring a reference dataset.

The simulator should support expression categories such as:

- Very low expression
- Low expression
- Moderate expression
- High expression

These categories are useful for evaluating method performance as a function of expression level.

A later version may support baseline parameters estimated from a real bulk RNA-seq dataset.

---

# 6. Differential-expression truth

The simulator should support both randomly generated and explicitly assigned differential-expression effects.

Recommended options include:

- Proportion of genes affected
- Explicit list of affected genes
- Separate proportions of upregulated and downregulated genes
- User-defined effect sizes for selected genes
- Effect-size distributions for automatically selected genes
- Different effects for different case conditions

Each gene should be labeled according to its truth status, for example:

- Null
- Differentially expressed in `case_1`
- Differentially expressed in `case_2`
- Differentially expressed in multiple conditions
- Affected by a confounder only
- Affected by both condition and confounder

Condition effects should be represented as **true log2 fold changes** relative to the reference condition.

Effects should be generated before count sampling and stored in the gene metadata so that downstream estimates can be compared with the known truth.

---

# 7. Dispersion model

Dispersion should be **gene-specific** and preferably related to mean expression.

The recommended initial approach is:

- Generate baseline expression for each gene
- Generate a mean-dependent dispersion
- Add controlled random variation around the dispersion trend
- Store the true dispersion in the gene metadata

This is more realistic than using one common dispersion and remains simpler than fully reproducing the estimation procedures used by DESeq2 or edgeR.

The simulator should allow users to control the overall dispersion level. This is important for power simulations because larger dispersion generally reduces the ability to detect differential expression.

---

# 8. Library-size variation

The simulator should include sample-specific library-size factors.

These factors represent variation in sequencing depth and allow downstream methods to be tested under realistic normalization requirements.

Recommended behavior:

- Generate library-size factors from a configurable log-normal distribution
- Center factors around one
- Store the true factor in sample metadata
- Allow users to provide custom factors if needed

The initial model should not force every sample to have exactly the same total count. Counts will be generated independently using the sample-specific expected means and library-size factors.

A later version could support fixed-depth or compositional sampling if this becomes important for a specific analysis.

---

# 9. Gene metadata

The gene metadata table should contain one row per gene and should align exactly with the rows of the count matrix.

Recommended fields include:

- Gene ID
- Gene index
- Baseline expression parameter
- Baseline expected count
- Expression category
- True dispersion
- Differential-expression status
- Condition-specific true log2 fold changes
- Confounder-specific effects
- Expected mean expression under each condition
- Expected mean expression under selected confounder settings
- Upregulated/downregulated label
- Optional gene-set or pathway membership

The metadata should distinguish between:

- **Biological effects of interest**, such as condition effects
- **Technical or unwanted variation**, such as batch effects
- **Continuous covariate effects**, such as age effects

This distinction is essential when evaluating confounder adjustment.

---

# 10. Sample metadata

The sample metadata table should contain one row per sample and should align exactly with the columns of the count matrix.

Recommended fields include:

- Sample ID
- Condition
- Condition index
- Batch
- Gender
- Age
- Standardized age
- Library-size factor
- Simulated total count
- Reference-condition indicator
- Optional replicate or subject ID
- Optional simulation identifier

The simulator should validate that:

- Every sample ID is unique
- Every count-matrix column has matching metadata
- Every condition has the requested number of samples
- Confounder values are valid
- Categorical reference levels exist
- Continuous variables contain valid numeric values

---

# 11. Output objects

The core simulation result should contain at least:

- Count matrix
- Gene metadata
- Sample metadata

The recommended design is to return a structured result object rather than three unrelated DataFrames. The result should also contain:

- Simulation configuration
- Random seed
- Diagnostic summaries
- Optional design matrix
- Optional expected-mean matrix

The count matrix should use this orientation:

- Rows: genes
- Columns: samples
- Values: non-negative integer counts

The three primary outputs should be exportable to common formats such as CSV, TSV, or Parquet. File-writing functionality should ideally be separate from the core simulation function.

---

# 12. Reproducibility

Reproducibility is mandatory.

Each simulation should support a user-defined random seed. The result should preserve:

- Random seed
- Complete configuration
- Number of genes
- Condition structure
- Confounder settings
- Effect-size settings
- Dispersion settings
- Library-size settings
- Simulator version
- Generated truth parameters

The simulator should allow the same configuration and seed to reproduce the same data exactly.

---

# 13. Diagnostics and validation

The simulator should produce diagnostic summaries automatically.

Important diagnostics include:

- Number of genes
- Number of samples
- Samples per condition
- Condition-by-confounder tables
- Distribution of library-size factors
- Distribution of baseline expression
- Distribution of dispersions
- Number of truly differential genes
- Number of genes affected by each confounder
- Fraction of zero counts
- Total counts per sample
- Mean–variance relationship
- Observed versus expected fold changes
- Degree of condition–confounder association

The simulator should issue warnings when settings are problematic, such as:

- Complete confounding between condition and batch
- Extremely small group sizes
- Very low expected expression
- Excessive dispersion
- Unrealistically large effect sizes
- Continuous confounders with no variation
- Invalid reference levels

Complete confounding may be useful intentionally for stress testing, but it should be explicitly reported because some effects cannot be statistically separated in that design.

---

# 14. Pathway and gene-set structure

Pathway enrichment is a future use case, but it should influence the architecture now.

The initial simulator should allow optional gene-set metadata, but pathway-level coordinated effects can be implemented in a later phase.

For meaningful pathway benchmarking, the simulator should eventually support:

- User-defined gene sets
- Coordinated upregulation
- Coordinated downregulation
- Null pathways
- Overlapping gene sets
- Different proportions of affected genes within a pathway

Randomly assigning differential genes does not create meaningful pathway-level signal. Therefore, pathway effects should be added deliberately rather than inferred from random gene-level effects.

---

# 15. Features intentionally excluded from the first version

To keep the simulator interpretable and manageable, the first version should exclude:

- Single-cell RNA-seq features
- Transcript-level sequence simulation
- Read-level simulation
- Alternative splicing
- Zero-inflated models
- Complex gene–gene correlation structures
- Subject-level repeated measures
- Paired samples
- Random effects
- Condition-by-confounder interactions
- Time-course designs
- Multiple nested study sites
- Technical sequencing artifacts

These features may be added later if they are required for a specific benchmarking objective.

---

# 16. Recommended development phases

## Phase 1: Core bulk RNA-seq simulation

Implement:

- Multiple conditions
- Unequal sample sizes
- Negative-binomial counts
- Gene-specific baseline expression
- Gene-specific dispersion
- Condition-specific effects
- Sample-specific library-size factors
- Reproducible random seeds
- Count matrix
- Gene metadata
- Sample metadata
- Basic diagnostics

## Phase 2: Confounding

Add:

- Two categorical confounders
- One continuous confounder
- Balanced assignment
- Random assignment
- Partial confounding
- Strong confounding
- User-defined confounder values
- Confounder effect sizes
- Confounding diagnostics

## Phase 3: Power-analysis support

Add:

- Repeated simulation workflows
- Power summaries across sample sizes
- Power summaries across effect sizes
- Power summaries across dispersion levels
- Detection performance by expression category
- False-positive and false-discovery summaries

## Phase 4: Structured biological signals

Add:

- Gene-set effects
- Coordinated pathway-level effects
- Optional gene–gene correlation
- More realistic empirical baseline distributions
- Reference-data calibration

## Phase 5: Advanced designs

Consider:

- Paired samples
- Repeated measures
- Subject-level random effects
- Interactions
- Time courses
- Multiple study sites
- Compositional count generation

---

# 17. Recommended initial defaults

The initial implementation should use these defaults unless the user overrides them:

- **Model**: bulk RNA-seq negative-binomial counts
- **Conditions**: arbitrary named conditions with unequal sample sizes
- **Reference**: user-defined, defaulting to `control`
- **Expression scale**: log-scale mean model
- **Effects**: log2 fold changes
- **Dispersion**: gene-specific and mean-dependent
- **Library sizes**: variable across samples
- **Categorical confounders**: two
- **Continuous confounders**: one
- **Interactions**: excluded
- **Gene correlation**: excluded
- **Pathway effects**: optional, implemented after the core model
- **Outputs**: counts, gene metadata, sample metadata, configuration, and diagnostics
- **Reproducibility**: mandatory random seed
- **Primary use**: differential-expression benchmarking, power analysis, and confounder-adjustment testing

---

# 18. Package concepts to use as inspiration

The simulator can borrow high-impact concepts from existing tools without reproducing their full complexity.

Relevant ideas include:

- **DESeq2**: size factors, negative-binomial modeling, dispersion trends, and design formulas
- **edgeR**: mean–variance behavior and biological variability
- **powsimR**: power-analysis-oriented simulation
- **compcodeR**: benchmarking structure and performance summaries
- **polyester**: RNA-seq simulation concepts
- **Splatter**: configurable simulation architecture, although it is primarily designed for single-cell data

The simulator should not attempt to replicate complete package internals. Its purpose is to create transparent datasets with known truth that can be analyzed by those packages.

---

# 19. Final design principle

The simulator should be built around the following conceptual components:

1. **Simulation configuration**
2. **Experimental design**
3. **Sample metadata**
4. **Gene-level truth**
5. **Expected expression means**
6. **Negative-binomial count generation**
7. **Validation and diagnostics**
8. **Exportable outputs**

The most important requirement is that every generated count should be traceable to known simulation parameters. This will make it possible to determine whether a downstream method failed because of low power, dispersion, confounding, normalization, or incorrect model specification.

# 20. Decision record: adopting powsimR as the simulation engine

## 20.1 Decision

We will use **powsimR** (R) as the core engine for count simulation and power
analysis, instead of building the custom simulator specified in Sections 1-19.
Python remains the orchestration layer (via `rpy2` or subprocess/CLI calls).
The custom design in this document is retained as a **fallback specification**.

## 20.2 Context

- Primary goals: power analysis, DE benchmarking, and (secondarily)
  confounder-adjustment testing.
- Audience: internal methods work.
- The analysis is a one-off, with thousands of replicates; computational cost
  is not a major concern.
- Confounders and a continuous covariate were rated as nice-to-have.
- Comfortable calling R from Python.

## 20.3 Why powsimR

- **Parameter estimation from real data**: mean, dispersion, and library-size
  distributions are estimated from a user-supplied count matrix, which reduces
  reliance on invented parametric assumptions.
- **Built-in power workflow**: simulation, DE testing, and power/FDR
  evaluation across sample sizes are integrated.
- **Ground truth is preserved**: DE status and log fold changes are injected
  by the simulator, so truth is known even though nuisance parameters come
  from real data.
- **Established method**: reduces the need to validate a custom generator.

## 20.4 Pros

- Much less development and validation effort than a custom simulator
- Realistic mean-dispersion structure from real data
- Direct support for power across sample sizes and effect sizes
- Wraps common DE tools, enabling method comparison

## 20.5 Cons and limitations

- **Two-group focus**: multi-condition designs (e.g., control, case_1, case_2)
  are not directly supported. Workaround: run each case versus control
  separately.
- **Limited confounder support**: designed batch/gender/age confounding with
  per-gene confounder truth is not provided out of the box.
- **R dependency**: requires R, Bioconductor, and interop with Python.
- **Output reshaping**: results are R objects and must be converted to the
  three target tables (counts, gene metadata, sample metadata).
- **Version sensitivity**: the API has changed across releases; pin versions.
- **Independent genes**: no gene-gene correlation, which can make FDR look
  optimistic compared with real data.

## 20.6 Triggers to revisit this decision

Reconsider a custom simulator (Sections 1-19) if any of the following
becomes a hard requirement:

1. Multi-condition designs analyzed jointly
2. Designed confounding with controlled association strength
3. A continuous covariate with a known per-gene effect
4. Composition effects or gene-gene correlation

## 20.7 Validation checklist

- Confirm null simulations give a false-positive rate near the nominal level
- Confirm injected log fold changes are recovered on average
- Compare simulated versus real mean-dispersion plots
- Record package versions and random seeds for every run

## Example code: real data to power analysis in powsimR

This follows the standard powsimR workflow: **estimate parameters, define a simulation setup, run the simulation and DE testing, then evaluate power**. Argument names may differ in your version, so check `?estimateParam`, `?Setup`, `?simulateDE`, and `?evaluateDE`.

```r
# --- Install (run once) -----------------------------------------------
# install.packages("devtools")
# BiocManager::install(c("DESeq2", "edgeR", "limma"))
# devtools::install_github("bvieth/powsimR")

library(powsimR)

set.seed(42)

# --- 1. Load real data ------------------------------------------------
# counts: genes x samples integer matrix; rownames = gene IDs
counts <- as.matrix(read.csv("real_counts.csv", row.names = 1))

# Basic filtering: remove genes that are almost never expressed
keep   <- rowSums(counts >= 5) >= 3
counts <- counts[keep, ]

# --- 2. Estimate parameters from real data ----------------------------
est <- estimateParam(
  countData     = counts,
  readData      = NULL,
  Lengths       = NULL,
  MeanFragLengths = NULL,
  RNAseq        = "bulk",
  Protocol      = "Read",
  Distribution  = "NB",
  Normalisation = "MR",     # DESeq2 median-ratio normalization
  GeneFilter    = 0.1,
  SampleFilter  = 3,
  sigma         = 1.96,
  NCores        = 1
)

plotParam(est, annotate = TRUE)   # check mean-dispersion fit

# --- 3. Define simulation setup ---------------------------------------
# Effect sizes: log2 fold changes drawn from a distribution
lfc_fun <- function(x) rnorm(x, mean = 0, sd = 1.5)

setup <- Setup(
  ngenes   = 5000,
  nsims    = 25,              # increase for final run (e.g., 1000+)
  p.DE     = 0.1,             # proportion of truly DE genes
  pLFC     = lfc_fun,
  p.G      = 1,
  p.B      = NULL,            # batch effects: not used here
  bLFC     = NULL,
  sim.seed = 42
)

# --- 4. Simulate data and run DE testing ------------------------------
# Compare sample sizes per group
sim <- simulateDE(
  n1           = c(3, 6, 10),
  n2           = c(3, 6, 10),
  sim.settings = setup,
  DEmethod     = "DESeq2",
  Normalisation = "MR",
  Label        = "power_demo",
  DropGenes    = FALSE,
  Preclust     = FALSE,
  NCores       = 1,
  verbose      = TRUE
)

# --- 5. Evaluate power and error rates --------------------------------
evaluation <- evaluateDE(
  simRes         = sim,
  alpha.type     = "adjusted",
  MTC            = "BH",
  alpha.nominal  = 0.05,
  stratify.by    = "mean",     # performance by expression level
  filter.by      = "none",
  strata.filtered = 1,
  target.by      = "lfc",
  delta          = 0
)

plotEvalDE(evaluation, rate = "marginal", quick = FALSE, annotate = TRUE)
plotEvalDE(evaluation, rate = "stratified", quick = FALSE, annotate = TRUE)
```
