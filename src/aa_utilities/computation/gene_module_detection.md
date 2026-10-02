# Brainstorm Continuation Document: Recursive Multiscale Gene Co-expression Modules for Drug-Response Prediction

> **Purpose:** This document summarizes the proposed method, key decisions, methodological alternatives, unresolved questions, and recommended next steps so the discussion can continue in a new session.

## Executive Summary

The project aims to use **baseline RNA-seq data from a clinical trial** to identify coherent gene co-expression modules that collectively predict whether patients will respond or not respond to a drug.

The proposed workflow is:

1. Preprocess baseline RNA-seq data.
2. Construct a **signed gene-gene co-expression network**, initially using Spearman correlations.
3. Start with all genes in a root cluster.
4. Recursively split clusters into smaller gene modules.
5. Stop splitting when modules are sufficiently coherent, stable, or small.
6. Summarize every module using a representative feature, primarily its **first principal component**, or module eigengene.
7. Use all or selected module features in a **regularized logistic regression model** to predict binary response.
8. Evaluate module stability, predictive performance, clinical added value, and biological interpretation.

The closest established methods are:

- **WGCNA**: the main baseline for weighted gene co-expression networks and module eigengenes.
- **MEGENA**: likely the closest conceptual neighbor because it identifies hierarchical, multiscale modules through network filtering and recursive partitioning.
- **Dynamic Tree Cut**: relevant for adaptive module-boundary selection.
- **Tree-guided group lasso and related structured-regularization methods**: relevant for incorporating the module hierarchy into prediction.

The central methodological recommendation is:

> **Use PC1 as a compact module representation, but do not make “PC1 explains more than 90% of variance” the sole stopping criterion.** Combine it with split quality, minimum size, and resampling stability.

---

# 1. Scientific Objective

## 1.1 Data structure

The planned data are:

- Baseline RNA-seq expression data.
- Samples collected from patients enrolled in a clinical trial.
- Patients classified at the end of the trial as responders or nonresponders.
- Expression measured before treatment or before the relevant treatment-response period.

## 1.2 Main objective

The goal is to discover **baseline transcriptional programs** that help predict response to treatment.

The intended interpretation is not that one module will fully explain response. Instead:

- Each module should represent a coherent expression program.
- Each module is reduced to one or a small number of representative features.
- Several modules may jointly contribute to prediction.
- Logistic regression should identify modules that collectively provide predictive value.

## 1.3 Important terminology

If “WGCA” refers to **Weighted Gene Co-expression Network Analysis**, the standard abbreviation is **WGCNA**.

The proposed custom method could eventually be described as:

> **Stability-validated recursive multiscale signed gene co-expression analysis for response prediction.**

This is a descriptive name, not an established method name.

---

# 2. Proposed Core Method

## 2.1 Root-to-leaf recursive design

The proposed method is **divisive**, or top-down:

1. Begin with all retained genes in a root cluster.
2. Evaluate whether the cluster should be split.
3. Find the best candidate split.
4. Accept the split only if it satisfies predefined criteria.
5. Repeat recursively on each child cluster.
6. Stop when further splitting is not justified.

This differs from standard WGCNA, which typically uses an **agglomerative**, or bottom-up, hierarchical clustering workflow.

## 2.2 Signed correlations

The agreed direction is to use **signed correlations** because positively and negatively co-expressed genes may represent distinct biological programs.

A natural signed distance is:

```math
d_{ij}=1-\rho_{ij}
```

where $$\rho_{ij}$$ is the correlation between genes $$i$$ and $$j$$.

Interpretation:

- $$\rho_{ij}=1$$: distance is 0; genes behave similarly.
- $$\rho_{ij}=0$$: intermediate distance.
- $$\rho_{ij}=-1$$: distance is 2; genes behave oppositely.

Using signed correlations means that genes moving in opposite directions are less likely to be placed in the same module.

## 2.3 Module representation using PCA

For each module, calculate the first principal component:

```math
ME_C = PC1(X_C)
```

where:

- $$C$$ is a gene module.
- $$X_C$$ is the expression matrix for the genes in that module.
- $$ME_C$$ is the module eigengene or PC1 score across patients.

The intended rationale is:

- Genes within a module should coherently increase or decrease across samples.
- Their shared signal can be represented by one feature.
- Dimensionality is reduced from thousands of genes to a much smaller number of module features.
- Logistic regression can then combine these module-level signals.

## 2.4 Orienting module eigengenes

The sign of a PCA score is arbitrary: a software implementation may return either $$ME$$ or $$-ME$$.

For consistent interpretation, orient each module eigengene against the average standardized expression of its genes:

```math
\text{If } \operatorname{cor}(ME,\bar{X}_{module})<0,
\quad
ME\leftarrow -ME
```

This ensures that a positive module score generally corresponds to higher average expression of the module genes.

---

# 3. Stopping and Splitting Rules

## 3.1 Original proposed stopping rule

The original idea was:

- Stop splitting if PC1 explains more than 90% of the variance within a cluster.
- Continue splitting otherwise.
- Stop when a cluster contains fewer than a threshold number of genes, such as 10.

## 3.2 Concern about the 90% rule

The 90% threshold is intuitive but should not be treated as a universal rule.

PC1 variance explained measures **internal dimensionality**, not necessarily biological coherence or predictive usefulness.

A high PC1 proportion may result from:

- A genuine shared transcriptional program.
- Batch effects.
- Cell-type composition.
- RNA quality.
- Technical variation.
- A general expression gradient.
- Mean-variance artifacts.

A response-relevant module can also have PC1 below 90% if:

- Only a subset of genes responds.
- The response is heterogeneous.
- The relevant signal is partly represented by PC2 or other components.
- Several related processes are present within the cluster.

## 3.3 Recommended stopping criteria

Use PC1 variance explained as one diagnostic, together with:

- **Minimum module size**: for example, 10 genes, subject to validation.
- **Within-module cohesion**: average signed correlation or another cohesion measure.
- **Between-child separation**: children should be more distinct from each other than the unsplit parent.
- **Split improvement**: the split must improve a prespecified objective.
- **Resampling stability**: the split should be reproducible under bootstrap or subsampling.
- **Minimum sample-effective complexity**: avoid creating more module features than the cohort can support.
- **Maximum tree depth**, if needed.
- **Optional PC1 threshold**, used descriptively or as a soft rule.

A more robust rule is:

> Stop splitting when the proposed split does not produce a meaningful, stable improvement in module separation and coherence, or when either child is too small.

## 3.4 Potential split-quality objective

A generic split score could be:

```math
Q(C_1,C_2)
=
\operatorname{Coh}(C_1)
+
\operatorname{Coh}(C_2)
-
\lambda\operatorname{CrossSim}(C_1,C_2)
```

where:

- $$C_1$$ and $$C_2$$ are candidate child modules.
- $$\operatorname{Coh}(C_k)$$ measures within-child cohesion.
- $$\operatorname{CrossSim}(C_1,C_2)$$ measures similarity across children.
- $$\lambda$$ controls the penalty for cross-child similarity.

The split should only be accepted if it improves the objective by a prespecified amount and is stable.

---

# 4. Candidate Splitting Methods

## 4.1 Average-linkage hierarchical splitting

### Concept

Perform average-linkage hierarchical clustering within the current cluster, then select a two-way cut.

### Advantages

- Familiar and straightforward.
- Easy to visualize.
- Computationally practical.
- Produces a natural hierarchy.
- Provides a useful baseline.

### Limitations

- Average linkage is a linkage rule, not a complete split-quality criterion.
- Early splitting errors are difficult to undo.
- Results depend on the dendrogram and cut-selection rule.
- May not fully use network topology.
- Can create a binary split even when the structure is weak.

### Recommendation

Use this as the first baseline implementation, but add explicit split-acceptance criteria.

---

## 4.2 Spectral bipartitioning

### Concept

Use the eigenvectors of a graph-derived matrix, such as a graph Laplacian, to divide the network into two groups.

### Advantages

- Naturally suited to graph-based co-expression data.
- Can detect complex, non-spherical structures.
- Provides a principled topological partition.
- Appropriate for recursive divisive clustering.

### Limitations

- Sensitive to graph construction and sparsification.
- Can be computationally demanding.
- May behave poorly with disconnected or weakly connected graphs.
- The mathematically optimal partition may not be biologically optimal.

### Recommendation

This is a strong alternative to average linkage and should be considered in a method comparison.

---

## 4.3 Two-medoids clustering

### Concept

Partition genes into two groups around two representative genes, called medoids.

### Advantages

- Medoids are actual genes and can be interpretable.
- Can use arbitrary distance measures.
- More robust to outliers than k-means.
- Relatively easy to understand.

### Limitations

- Assumes a two-cluster split at each step.
- Can depend on initialization.
- May favor compact expression-space clusters rather than network modules.
- Still requires a rule for accepting the split.
- Can become computationally expensive for large clusters.

### Recommendation

Useful as a simple alternative, particularly if representative genes are desired.

---

## 4.4 Graph-cut methods

### Concept

Partition a weighted co-expression graph while minimizing cross-cluster connectivity and preserving strong within-cluster connectivity.

### Advantages

- Directly models network structure.
- Can optimize separation and internal connectivity.
- Flexible for sparse or weighted graphs.
- Provides an explicit partition objective.

### Limitations

- Sensitive to graph density and edge thresholds.
- Dense or noisy networks can generate arbitrary cuts.
- Some versions are computationally intensive.
- Mathematical separation does not guarantee biological interpretability.

### Recommendation

Attractive for a network-oriented version of the method, especially if normalized-cut-type objectives are used.

---

## 4.5 MEGENA-style recursive network clustering

### Concept

Use a filtered co-expression network followed by multiscale recursive clustering.

### Advantages

- Specifically designed for multiscale gene co-expression.
- Produces nested modules.
- Uses network topology rather than only pairwise expression distance.
- Provides an established benchmark.

### Limitations

- More complex than the proposed custom method.
- Includes several implementation and tuning choices.
- Planar-network assumptions may not suit every dataset.
- Less minimal and less transparent for an initial prototype.

### Recommendation

Use MEGENA as a benchmark and conceptual comparator rather than necessarily reproducing the entire method.

---

# 5. MEGENA Network Filtering

The closest published method to the proposed recursive design is probably **MEGENA**, introduced by Song and Zhang in 2015.

Its network-filtering component is called **Fast Planar Filtered Network Construction**, or FPFNC.

## 5.1 Basic process

MEGENA begins with gene-gene similarities, usually derived from expression correlations.

The initial network is very dense because the number of possible gene pairs is:

```math
\frac{p(p-1)}{2}
```

where $$p$$ is the number of genes.

MEGENA reduces this dense network by:

1. Computing gene-gene similarities.
2. Filtering candidate edges using statistical criteria such as FDR.
3. Ranking candidate edges by strength.
4. Adding strong edges while maintaining a planar-network constraint.
5. Using parallelized planarity checks.
6. Using early termination to avoid processing increasingly weak edges.

The result is a sparse network backbone intended to preserve strong, topologically informative relationships.

## 5.2 Why this may reduce noise

Filtering can:

- Remove weakly supported associations.
- Reduce redundant edges.
- Make network structure easier to detect.
- Reduce computational cost.
- Highlight strong local connectivity.

It does not guarantee that all retained edges are biological or causal, and removed edges are not necessarily false.

## 5.3 Applicability to the proposed method

A simpler custom alternative would be:

- Calculate signed Spearman correlations.
- Apply a prespecified strength and/or statistical filter.
- Build a sparse signed graph.
- Use that graph for recursive splitting.

A key caution is that a fixed threshold such as $$|\rho|>0.3$$ behaves differently for different sample sizes. Network construction should therefore be assessed through stability analysis.

---

# 6. Correlation Choices

## 6.1 Spearman correlation

Spearman correlation measures the correlation between ranked expression values.

### Advantages

- Robust to outliers compared with Pearson correlation.
- Captures monotonic, not necessarily linear, relationships.
- Easy to interpret.
- Consistent with the original proposal.

### Limitations

- Discards magnitude information by using ranks.
- Can be unstable with very small sample sizes.
- Ties are common in low-expression RNA-seq data if preprocessing is inadequate.

## 6.2 Biweight midcorrelation

**Biweight midcorrelation**, or **bicor**, is a robust correlation measure based on medians, median absolute deviations, and smooth observation weights.

It downweights observations that are far from the center of a gene’s expression distribution.

Conceptually:

```math
w_i =
\begin{cases}
(1-u_i^2)^2, & |u_i|<1 \\
0, & |u_i|\geq 1
\end{cases}
```

where $$u_i$$ measures the scaled distance from the median.

### Advantages

- Less sensitive to extreme observations.
- Retains continuous expression magnitude.
- Often useful for co-expression analysis.

### Limitations

- May downweight rare but biologically important patient subgroups.
- Can behave poorly with very small sample sizes.
- Requires choices about outlier handling.

### Recommendation

Keep signed Spearman as the primary approach initially and use bicor as a sensitivity analysis.

## 6.3 Shrinkage or regularized correlation

Raw correlation estimates can be noisy when the number of genes is large relative to the number of samples.

Shrinkage combines the empirical estimate with a stable target:

```math
\widehat{\Sigma}_{shrunk}
=
(1-\lambda)S+\lambda T
```

where:

- $$S$$ is the empirical covariance matrix.
- $$T$$ is a structured target.
- $$\lambda$$ is the shrinkage intensity.

This can stabilize covariance or correlation estimates.

### Relevant approaches

- **Ledoit-Wolf shrinkage**
- **Schäfer-Strimmer shrinkage**
- **Graphical lasso**, which estimates sparse conditional relationships through a precision matrix

### Recommendation

Do not add shrinkage unless it is needed. Prioritize preprocessing, signed correlations, and stability analysis first. Shrinkage becomes more attractive when:

- The cohort is small.
- Correlation estimates are unstable.
- Partial correlations are being used.
- A sparse precision network is desired.

---

# 7. Parent and Child Module Features

## 7.1 The proposed principle

The intended reasoning is:

> If a parent module predicts response, the information should also be present in its child modules, and logistic regression should be able to identify a predictive combination of the children.

This is **conceptually valid under certain conditions**, but not guaranteed for PCA-compressed child features.

## 7.2 When the principle holds exactly

If a parent score is explicitly defined as a weighted combination of child scores:

```math
M_P=w_1M_{C_1}+w_2M_{C_2}
```

then a logistic regression model using the child scores can represent the same parent effect:

```math
\operatorname{logit}\{P(Y=1)\}
=
\beta_0+\beta_PM_P
```

becomes:

```math
\operatorname{logit}\{P(Y=1)\}
=
\beta_0+\beta_Pw_1M_{C_1}+\beta_Pw_2M_{C_2}
```

## 7.3 Why it is not guaranteed with separate PC1 scores

The parent PC1 is calculated from all parent genes, while each child PC1 is calculated independently.

In general:

```math
PC1(X_P)
\neq
a_1PC1(X_{C_1})+a_2PC1(X_{C_2})
```

because:

- Child PC2 or later components may contain relevant parent information.
- The parent PC1 may depend on gene loadings not preserved by child PC1.
- Child PC1 scores are individually optimized, not jointly optimized to reconstruct the parent.
- Opposite-direction child signals may require contrasts.
- The children may be highly correlated.
- Penalized logistic regression may select one proxy and discard another.

Thus, the information may remain in the full child gene expression matrices while being lost in the one-PC-per-child representation.

## 7.4 Recommended empirical test

For each parent module:

1. Calculate the parent PC1.
2. Calculate the child PC1 scores.
3. Regress the parent PC1 on its child PC1 scores.
4. Calculate the reconstruction quality, such as:

```math
R^2_{\text{children}\rightarrow\text{parent}}
```

If the reconstruction is high, the child summaries preserve the parent signal well.

If it is low, compare:

- Parent-only features.
- Child-only features.
- Parent-plus-child features.
- Child PC1 plus additional child PCs.
- Child residual features after removing parent information.

## 7.5 Parent-plus-child and residual representations

A useful decomposition is:

```math
M_C^{residual}
=
M_C-\widehat{E}(M_C\mid M_P)
```

This represents the child-specific signal not already captured by the parent.

A model can then use:

- The parent score for broad program activation.
- Child residual scores for more specific deviations.

This may be more stable and interpretable than asking a penalized model to choose between highly correlated parent and child eigengenes.

## 7.6 Recommended practical position

The working principle should be:

> **Descendant genes should contain the parent’s information, but descendant PC1 features may not preserve all of it.**

Therefore, include internal and terminal modules in exploratory analyses, but evaluate parent-versus-child representations empirically rather than assuming perfect equivalence.

---

# 8. Prediction Model

The intended outcome is binary response.

A basic logistic model is:

```math
\operatorname{logit}\{P(Y=1)\}
=
\beta_0+\sum_{k=1}^{K}\beta_kME_k+\gamma^\top Z
```

where:

- $$Y$$ is responder status.
- $$ME_k$$ is a module feature.
- $$Z$$ contains clinical covariates.
- $$\beta_k$$ measures the contribution of module $$k$$.
- $$\gamma$$ represents clinical covariate effects.

Because modules may be numerous and correlated, consider:

- Elastic-net logistic regression.
- Sparse group lasso.
- Hierarchical group lasso.
- Tree-guided group lasso.
- Parent-child residualization.
- Dimension reduction across highly correlated module eigengenes.

## 8.1 Avoiding information leakage

If response labels are used to select modules or tune the prediction model, the following should occur inside training folds:

- Gene filtering, if outcome-informed.
- Correlation estimation, if outcome-informed.
- Module construction, if outcome-informed.
- Module selection.
- Model fitting.
- Hyperparameter tuning.

The held-out test fold must remain untouched until final evaluation.

Even if module discovery is unsupervised, reconstructing modules within training folds is preferable when assessing the full predictive pipeline.

## 8.2 Compare against useful baselines

Compare:

- Clinical covariates only.
- Individual-gene model.
- Standard WGCNA modules.
- MEGENA modules.
- Proposed recursive modules.
- Predefined pathway scores.
- Proposed modules plus clinical covariates.

---

# 9. Validation and Stability

## 9.1 Predictive metrics

Report more than a single AUC:

- Cross-validated ROC AUC.
- Precision-recall AUC, especially if response is uncommon.
- Balanced accuracy.
- Sensitivity and specificity.
- Calibration.
- Brier score.
- Confidence intervals.
- Performance compared with a clinical-only model.

## 9.2 Module stability

Assess whether modules are reproducible through:

- Bootstrap resampling.
- Repeated subsampling.
- Alternative correlation estimators.
- Alternative split methods.
- Alternative gene-filtering thresholds.
- Alternative minimum module sizes.

Record:

- Module preservation.
- Gene membership stability.
- Split stability.
- Module-score correlation across resamples.
- Selection frequency in predictive models.

## 9.3 Treatment-arm interpretation

If all patients received the drug, the resulting signature is associated with or predictive of response **among treated patients**.

It cannot establish treatment-effect prediction without a comparator group.

If a control arm exists, treatment-predictive behavior should be tested using an interaction model:

```math
\operatorname{logit}\{P(Y=1)\}
=
\beta_0+\beta_1T+\beta_2M+\beta_3(T\times M)+\gamma^\top Z
```

The interaction coefficient $$\beta_3$$ assesses whether the molecular score modifies treatment response.

---

# 10. Preprocessing Considerations

Recommended preprocessing includes:

- Filter very lowly expressed genes.
- Normalize the RNA-seq data appropriately.
- Use a variance-stabilizing transformation or suitable log-scale transformation.
- Inspect sample-level PCA and clustering.
- Identify technical outliers.
- Account for batch, site, RNA quality, and library-preparation effects.
- Consider filtering to a suitable number of variable genes.

Potential confounders include:

- Sequencing batch.
- RNA integrity.
- Library size.
- Processing site.
- Sample collection date.
- Sex.
- Disease severity.
- Cell-type composition.

Cell-type composition requires special care. It could be:

- A technical or biological confounder to adjust for.
- A mediator of response.
- A clinically useful predictive signal.

Do not automatically regress it out without deciding which interpretation is scientifically desired.

---

# 11. Closest Relevant Publications and Methods

## 11.1 WGCNA

Key contributions:

- Weighted gene co-expression networks.
- Hierarchical clustering on topological overlap.
- Module eigengenes based on PC1.
- Module-trait association analysis.

Important clarification:

> WGCNA does not require PC1 to explain 90% of module variance. PC1 is primarily used to summarize a module.

## 11.2 MEGENA

The closest conceptual neighbor.

Relevant features:

- Network filtering.
- Multiscale module identification.
- Recursive or divisive module organization.
- Nested modules.
- Module-trait association.

## 11.3 Dynamic Tree Cut

Relevant for adaptive dendrogram cutting and module-boundary selection.

## 11.4 Tree-guided group lasso

Relevant for incorporating hierarchical structure into a downstream predictive model.

## 11.5 Other useful comparators

- CEMiTool.
- Consensus WGCNA.
- GSVA.
- ssGSEA.
- Sparse group lasso.
- Graph-guided fused lasso.
- PLIER.
- Pathway-based response models.

The most important initial benchmark set is:

> **Proposed recursive method versus WGCNA versus MEGENA.**

---

# 12. Recommended Initial Analysis Plan

## Phase 1: Method development

Develop a response-independent module hierarchy using:

- Baseline expression only.
- Signed Spearman correlation.
- Average-linkage recursive splitting.
- Minimum module size.
- Split-quality assessment.
- Bootstrap stability.
- PC1 module summaries.

## Phase 2: Method comparison

Compare:

- Average-linkage splitting.
- Spectral bipartitioning.
- MEGENA.
- WGCNA.

Compare them by:

- Module stability.
- Number and size of modules.
- PC1 coherence.
- Biological enrichment.
- Redundancy among module scores.
- Predictive performance under nested cross-validation.

## Phase 3: Predictive modeling

Fit:

- Clinical-only logistic regression.
- Module-only logistic regression.
- Clinical-plus-module logistic regression.
- Elastic-net logistic regression.
- Hierarchical or tree-guided models, if justified.

## Phase 4: Parent-child analysis

For each parent-child relationship:

- Assess child-to-parent PC1 reconstruction.
- Compare parent-only and child-only predictors.
- Test parent-plus-child features.
- Consider child residual features.
- Evaluate coefficient and module-selection stability.

## Phase 5: Validation and interpretation

- Perform nested cross-validation.
- Evaluate module selection frequency.
- Annotate selected modules.
- Check pathway and cell-type enrichment.
- Assess external validation if available.
- Freeze the final signature before independent testing.

---

# 13. Open Questions for the Next Session

The following details will determine the most appropriate implementation:

- How many total patients are available?
- How many are responders and nonresponders?
- Is there a randomized control arm?
- Are all patients receiving the same treatment?
- What is the sample type: tumor, blood, tissue, biopsy, or another source?
- Are the RNA-seq data raw counts, normalized counts, or transformed expression?
- Are there multiple trial sites or batches?
- Is the response endpoint binary by design, or derived from a continuous or time-to-event endpoint?
- How many genes are expected to remain after filtering?
- Is the primary goal biological discovery, prediction, or both?
- Is external validation available?
- Should modules be built globally across all patients, or separately within treatment arms or clinical subgroups?

---

# 14. Recommended Next Step

The most useful next step is to create a **formal method specification** before coding.

That specification should define:

- Input data format.
- Expression preprocessing.
- Correlation estimator.
- Signed distance.
- Candidate splitting algorithm.
- Split-quality metric.
- Stopping rules.
- Module-score definitions.
- Parent-child feature handling.
- Logistic-regression strategy.
- Cross-validation scheme.
- Stability metrics.
- Primary and secondary evaluation criteria.

Once these decisions are fixed, the method can be implemented and tested first on simulated data with known module structure, followed by a pilot analysis of the clinical-trial dataset.