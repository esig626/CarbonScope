# CarbonScope scientific workflow

CarbonScope is intended to use isotope tracing as a forward hypothesis testing framework, not primarily as a tool for recovering one supposedly true flux vector.

This document describes the intended scientific workflow and distinguishes it from the implemented finite testing scope. The current release provides forward modelling, feasible state sampling, genuine count observation laws, simple binary testing and composite testing for represented finite H0/H1 classes. The public `run_hypothesis_testing_workflow(...)` API and `fluxemu test-hypotheses` command now construct those classes from one common SBML/FBC model and explicit hypothesis and experiment files. Testing over a complete continuous feasible flux family remains unsolved in this implementation: rigorous optimisation or bounds over that entire family are not provided.

See [HYPOTHESIS_WORKFLOW.md](HYPOTHESIS_WORKFLOW.md) for a complete reproducible files-to-report example. The implemented workflow evaluates a declared experiment's finite-class testing performance. It does not accept observed data to produce a generic composite p-value or invert a test into compatibility regions.

The central question is:

> Given what was observed experimentally, which biological possibilities remain compatible with the data, and which competing possibilities can the experiment actually distinguish?

## 1. Formulate a biological hypothesis

The experimentalist proposes a metabolic network hypothesis describing the biological mechanism they believe could generate the observed system.

This hypothesis should define a family of biologically admissible metabolic states rather than a single preferred flux vector.

The V1 workflow requires separate H0 and H1 reaction-bound constraints on the same physical model. Restrictions intersect the original bounds; equal lower and upper bounds fix a reaction, and a zero interval excludes its flux. Hypotheses are never inferred from reaction names, observed MIDs, sample clusters or FVA ranges. Overlap between the two regions is permitted, but declarations that define the same feasible region are rejected.

## 2. Define the feasible flux region

Run FBA to establish that the proposed network can support the required biological behaviour.

Run FVA to characterise the feasible flux region under the chosen constraints.

FVA extrema are diagnostics only. They are not assembled into flux states.

For the hypothesis workflow, each region contains all steady-state fluxes satisfying the common physical bounds and its declared reaction restrictions. No retained objective fraction is added. The native experiment file's `fva_fraction_of_optimum` remains a forward-analysis setting and does not define H0 or H1.

## 3. Sample complete feasible flux states

Sample complete jointly feasible flux states from the admissible region.

The sampled states provide a finite representation of the biological hypothesis family. When passed to the finite composite engine, those explicitly listed states define its uncertainty class. Sampling alone does not prove that the list covers the full feasible region or controls errors uniformly over states outside the list. State sampling frequencies are not priors or weights in the testing problem.

V1 uses the native hit-and-run sampler with explicit family sizes and seeds. It checks every returned complete state against its constrained model, preserves canonical reaction and state order, and records the policy, seed, requested/actual size and state identities. A finite chain does not establish independent samples, convergence or adequate mixing.

## 4. Push the hypothesis through the forward isotope model

For every sampled flux state, run the EMU forward model under the proposed tracer experiment.

This produces a family of possible observable MIDs implied by the biological hypothesis.

Conceptually:

```text
biological hypothesis
        |
        v
feasible flux family
        |
        v
sampled complete flux states
        |
        v
EMU forward model
        |
        v
family of possible observable MIDs
```

The scientific object of interest is the whole observable family, not the single member closest to the data.

## 5. Ask whether the experiment is informative before running it

If competing biological hypotheses are available, propagate each through the same forward workflow to obtain competing families of observable laws.

The finite composite layer compares explicitly supplied families of complete genuine-count laws. A law may be an ordered product of count blocks when their independence is declared explicitly. It provides order-specific Rényi converse bounds, achieved score tests with directly evaluated worst-case errors, and a complete finite observation-space minimax LP for small problems that pass its numerical checks.

If reliable discrimination is impossible at the proposed sample size, redesign the experiment before collecting data. Possible changes include the tracer, measured targets, biological constraints or sample size.

A converse lower bound can establish that a given count total is insufficient for the represented classes. Sufficiency requires an achieved testing procedure or a corresponding upper guarantee. The relevant ordering is `converse <= beta_star <= achieved Type II error` under the same Type I budget. A calibrated score test need not attain the unrestricted minimax optimum. The LP is a mathematically exact characterisation of the finite problem; its floating-point solution remains subject to numerical validation and explicit refusal. These finite-class results do not certify distinguishability over the complete continuous flux family.

The workflow report keeps six quantities separate: order-specific composite converse, unrestricted represented finite minimax, finite-family Rényi candidate score, verified analytical score bound, deterministic analytical-score achieved errors, and calibrated score-family achieved errors. Every requested procedure has an evaluated or refused outcome with its reason. A failed analytical moment certificate does not turn a candidate into a verified projected score; direct calibration may still be available for a well-defined candidate.

## 6. Perform the experiment

Collect experimental isotope tracing measurements under the same declared tracer and measurement design.

The observation model used for inference must match the semantics of the measurement. Genuine counts may support an explicit count law. Continuous corrected MIDs require a separately justified observation model.

## 7. Test the experimental data against the hypothesis family

Do not select the single simulated MID closest to the experimental data and call its flux vector the answer.

The intended broader workflow would test compatibility with the entire observable family implied by the biological hypothesis. The implemented composite engine supports a declared binary decision rule between two explicit finite classes, with uniform Type I and Type II errors over their supplied members. It does not provide a generic composite p-value, inversion into compatibility or confidence regions, or continuous-family testing.

For a supported finite comparison, specify the classes, measurement law and testing level before examining the data. Ordinary corrected MIDs, peak areas and percentages cannot supply multinomial count totals.

## 8. Interpret the result

The scenarios below describe the scientific interpretation sought from a separately justified compatibility procedure. A binary finite-class decision by itself does not establish that either whole biological family is compatible, identify all surviving families, or justify a confidence region. Any implemented error guarantee applies only to the declared laws and supplied finite class.

### The proposed family is compatible with the data

The biological hypothesis survives. This does not imply that one flux state within the family is uniquely true.

### One family is rejected and a competing family survives

The data support discrimination between the two biological explanations at the declared testing level.

### Multiple competing families survive

The experiment does not resolve the biological question. The remaining ambiguity is scientific information about the limits of the experiment, not a failure that should be hidden by reporting a single best fit.

### No proposed family is compatible with the data

The current biological picture is insufficient, or one of the modelling or measurement assumptions is wrong. Global rejection does not identify the responsible reaction, pathway or assumption.

## 9. Use MFA only as a diagnostic search tool after rejection

If the proposed family is rejected, MFA may be used to search the feasible model space for regions that reproduce the observations more closely.

The fitted flux map is not interpreted as the true biological state. Its role is to suggest where a new biologically meaningful hypothesis might live.

A new family constructed using the experimental data must not then be tested naively on those same data. That would make the hypothesis data dependent and invalidate the clean testing interpretation. The revised family requires new data, held out data or a testing procedure that explicitly accounts for the selection step.

## 10. Core principle

CarbonScope should preserve the uncertainty that the experiment genuinely leaves unresolved.

The objective is not:

```text
experimental MIDs -> one best flux vector
```

The objective is:

```text
biological hypothesis
        |
        v
family of feasible metabolic possibilities
        |
        v
family of predicted experimental outcomes
        |
        v
experimental data
        |
        v
which biological possibilities survive?
```

MFA can assist the search. The forward model and hypothesis testing framework provide the scientific inference.
