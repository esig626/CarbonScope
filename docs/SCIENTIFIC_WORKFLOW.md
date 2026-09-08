# CarbonScope scientific workflow

CarbonScope is intended to use isotope tracing as a forward hypothesis testing framework, not primarily as a tool for recovering one supposedly true flux vector.

This document describes the intended scientific workflow. The current release already provides the forward modelling, feasible state sampling, genuine count observation laws and simple binary testing pieces. Composite testing over whole hypothesis families is planned and is not yet implemented.

The central question is:

> Given what was observed experimentally, which biological possibilities remain compatible with the data, and which competing possibilities can the experiment actually distinguish?

## 1. Formulate a biological hypothesis

The experimentalist proposes a metabolic network hypothesis describing the biological mechanism they believe could generate the observed system.

This hypothesis should define a family of biologically admissible metabolic states rather than a single preferred flux vector.

## 2. Define the feasible flux region

Run FBA to establish that the proposed network can support the required biological behaviour.

Run FVA to characterise the feasible flux region under the chosen constraints.

FVA extrema are diagnostics only. They are not assembled into flux states.

## 3. Sample complete feasible flux states

Sample complete jointly feasible flux states from the admissible region.

The sampled states are a numerical representation of the biological hypothesis family. They are not individual scientific hypotheses and they are not estimates of the true flux state.

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

The planned composite testing layer will use finite sample Rényi bounds to ask whether the proposed experiment can distinguish those families at the intended sample size and error constraints.

If reliable discrimination is impossible at the proposed sample size, redesign the experiment before collecting data. Possible changes include the tracer, measured targets, biological constraints or sample size.

A converse bound can establish that a given sample size is insufficient. Sufficiency requires an achievable testing procedure or corresponding upper guarantee.

## 6. Perform the experiment

Collect experimental isotope tracing measurements under the same declared tracer and measurement design.

The observation model used for inference must match the semantics of the measurement. Genuine counts may support an explicit count law. Continuous corrected MIDs require a separately justified observation model.

## 7. Test the experimental data against the hypothesis family

Do not select the single simulated MID closest to the experimental data and call its flux vector the answer.

Instead, test whether the experimental observations are compatible with the entire family of observable outcomes implied by the biological hypothesis.

With two competing hypotheses, compare the corresponding families directly.

## 8. Interpret the result

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
