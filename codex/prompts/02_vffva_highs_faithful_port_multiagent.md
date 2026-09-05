# Codex task: faithful VFFVA algorithmic port to HiGHS

Repository: `esig626/fluxemu-standalone`

Work on branch: `codex/vffva-highs-faithful-port`

Base the branch on current `main` after Stage 1 completion.

Read `AGENTS.md` first and obey it exactly.

Then execute this prompt in full and **only this prompt**.

---

# 0. Mission and stopping condition

The task is narrowly defined:

> Replace the current multiprocessing-based FastFVA implementation with a faithful HiGHS-native translation of the shared-memory computational architecture used by Marouen Ben Guebila's VFFVA / `veryfastFVA`, while preserving FluxEMU's more general scientific semantics and correctness guarantees.

This is **not** a task to invent a new FVA algorithm.

This is **not** a task to compete conceptually with VFFVA.

This is **not** a task to redesign FluxEMU Stage 1, sampling, EMU, MFA, or any information-theoretic machinery.

The intended relationship is:

```text
VFFVA computational architecture
        |
        | faithful solver/backend translation
        v
FluxEMU + HiGHS
```

The implementation should be recognisably the same computational strategy as VFFVA:

```text
prepare/solve biological LP once
        -> impose retained objective once
        -> create one persistent solver copy per shared-memory worker
        -> each worker uses solver-internal threads = 1
        -> MAX pass over reactions with dynamic load balancing
        -> MIN pass over reactions with dynamic load balancing
        -> for each endpoint only change objective coefficient/sense, reoptimize same LP, read result, clear objective coefficient
        -> retain worker-local simplex state across repeated solves
        -> restore canonical reaction order
```

The task is complete only when:

1. the production fast FVA path uses a faithful shared-memory VFFVA-style worker architecture with HiGHS;
2. the current spawn/multiprocessing implementation is no longer the normal production FastFVA path;
3. correctness parity with the existing cold FluxEMU reference is preserved across the repository's relevant FVA cases;
4. direct benchmark infrastructure attempts and, where executable, performs comparison with the original upstream VFFVA implementation on the same LP/model semantics;
5. multi-worker scaling is measured on models large enough for scheduling overhead to amortize;
6. the implementation is documented as a HiGHS-native port/adaptation of VFFVA's computational strategy, not as a novel FastFVA algorithm;
7. every required CI gate is green;
8. all work is committed and pushed on `codex/vffva-highs-faithful-port`, with a PR opened against `main`;
9. the supervisor agent defined below explicitly signs off that the task was implemented faithfully and no agent took an easier substitute route.

Stop there.

---

# 1. Hard scope exclusions

Do **not** implement or modify unless a minimal compatibility fix is strictly required by the FVA port:

- feasible-state sampling algorithms;
- EMU graph/compiler semantics;
- stationary or transient EMU mathematics;
- MFA;
- parameter fitting;
- confidence intervals;
- profile likelihoods;
- Bayesian methods;
- hypothesis testing;
- Rényi/KL/information theory;
- biological analysis;
- atom mapping semantics;
- SBML scientific semantics;
- new solver backends other than HiGHS;
- MPI/OpenMPI as a FluxEMU runtime dependency;
- CPLEX as a FluxEMU runtime dependency;
- GLPK as a FluxEMU runtime dependency.

Do not broaden the task because another part of the repository looks improvable.

---

# 2. Authoritative upstream reference

Use upstream:

```text
https://github.com/marouenbg/VFFVA
```

Pin the audit to upstream `master` commit:

```text
7cf7b82505bf99aed38a2073e3ed308f79e95802
```

At minimum read closely:

```text
lib/veryfastFVA.c
lib/VFFVA.py
README.md
LICENSE.txt
```

The important VFFVA mechanics to preserve are the computational ones, especially the CPLEX path in `veryfastFVA.c`:

- one solver environment/problem clone per OpenMP worker;
- each solver clone uses one internal solver thread;
- one persistent LP clone is reused for many reaction endpoint optimisations;
- outer objective-sense passes: maxima then minima;
- objective coefficient changed only for the reaction currently optimised;
- LP reoptimised without rebuilding the model;
- objective coefficient cleared after each solve;
- OpenMP runtime scheduling provides dynamic load balancing;
- the Python wrapper defaults to dynamic scheduling with chunk size 50;
- result order is independent of completion order.

Do **not** copy the C source line-for-line into FluxEMU. Perform a faithful algorithmic translation and attribute VFFVA clearly in docs/source comments where appropriate.

The repository-level VFFVA licence says MIT while `veryfastFVA.c` carries a CC BY 4.0 header. Avoid unnecessary source-copy licensing ambiguity by translating the algorithm rather than mechanically copying code.

---

# 3. FluxEMU scientific semantics that MUST remain stronger than upstream VFFVA

A faithful computational port does **not** mean downgrading FluxEMU mathematics to VFFVA's simplifying assumptions.

Preserve current FluxEMU semantics for:

1. arbitrary declared linear biological objectives, including multi-reaction objectives;
2. maximizing and minimizing biological objectives;
3. exact canonical reaction ordering;
4. current retained-objective formulation;
5. fixed/blocked reactions;
6. signed/reversible reactions;
7. current numerical conditioning/validation contract;
8. immutable/canonical model identity and fingerprints where already defined.

Do not replace FluxEMU's retained-objective row with VFFVA's assumption that one objective reaction can simply have its bound changed.

For a maximizing objective preserve the effective scientific constraint corresponding to

```text
c^T v >= f z*
```

and for minimization preserve the corresponding upper retained-objective constraint.

Do not copy VFFVA's objective rounding behaviour if it would weaken FluxEMU's existing retained-objective semantics.

The correct design principle is:

> Port VFFVA's execution architecture; preserve FluxEMU's scientific model semantics.

---

# 4. NON-NEGOTIABLE SUPERVISOR AGENT

Before spawning any implementation specialist, create one persistent agent named:

```text
VFFVA-GUARDIAN
```

or an unmistakably equivalent name.

This agent is **not** a normal reviewer at the end. It is the task supervisor for the entire run.

## 4.1 Supervisor authority

The VFFVA-GUARDIAN has veto authority over every specialist plan and every milestone.

The lead agent must not accept a specialist result, start the next implementation milestone, or declare completion until the guardian has reviewed it.

Every specialist must report through the lead to the guardian at these points:

1. after its initial source audit and before substantial coding;
2. after it proposes an implementation design;
3. after the relevant diff/tests exist;
4. before its milestone is considered finished.

For each review, the guardian must return one of:

```text
APPROVE
REJECT - <specific corrective actions>
BLOCKED - <concrete reproduced blocker>
```

A vague approval is not sufficient.

## 4.2 Supervisor's primary job: prevent task drift and easy substitutions

The guardian must actively prevent agents from taking shortcuts such as:

- leaving multiprocessing as the production implementation because it already works;
- merely changing `chunksize` and calling that a VFFVA port;
- using `ProcessPoolExecutor` or another process abstraction and calling it shared-memory VFFVA;
- calling the external VFFVA executable from FluxEMU production instead of translating the algorithm to HiGHS;
- adding MPI because upstream has MPI, when this task is specifically the shared-memory HiGHS port;
- rebuilding a HiGHS LP for every endpoint;
- creating one HiGHS object per endpoint instead of one persistent object per worker;
- letting HiGHS itself use multiple threads inside each worker while also running worker-level parallelism;
- interleaving min/max tasks arbitrarily without first testing the faithful VFFVA max-pass/min-pass ordering;
- using endpoint-level process IPC when highspy releases the GIL for `Highs.run()`;
- claiming HiGHS is not thread-safe without a reproduced, isolated failure or authoritative upstream evidence;
- keeping expensive non-VFFVA work in the hot loop without profiling whether it destroys the performance architecture;
- weakening or deleting correctness tests to gain speed;
- changing tolerances simply to make parity pass;
- benchmarking only against FluxEMU's deliberately cold correctness oracle and presenting that as the relevant external result;
- benchmarking only the 95-reaction E. coli core model and drawing conclusions about parallel scaling;
- declaring the direct VFFVA comparison "optional future work" without first making a serious attempt to run the available upstream implementation/backend;
- expanding into sampling, MFA, information theory, or unrelated cleanup.

If any agent proposes one of these shortcuts, the guardian must reject it and send it back with specific corrective instructions.

## 4.3 Supervisor acceptance ledger

The guardian must maintain a live internal checklist with at least these items:

```text
[ ] upstream VFFVA source audited at pinned commit
[ ] exact VFFVA -> HiGHS operation mapping written down
[ ] highspy shared-memory/thread assumptions verified
[ ] one persistent Highs instance per worker
[ ] Highs internal threads fixed to 1 per worker
[ ] workers persist across max and min passes
[ ] max pass then min pass preserved
[ ] dynamic shared-memory scheduling implemented
[ ] default dynamic chunk size begins from VFFVA's 50
[ ] no per-endpoint LP reconstruction
[ ] no per-endpoint process spawn/IPC in production path
[ ] serial correctness parity
[ ] threaded correctness parity
[ ] minimization parity
[ ] multi-term objective parity
[ ] fixed/blocked/reversible parity
[ ] deterministic canonical output ordering
[ ] contextual endpoint failure propagation
[ ] hot-loop validation costs profiled and justified
[ ] small-model overhead benchmark
[ ] medium/large-model scaling benchmark
[ ] direct original-VFFVA benchmark attempted
[ ] direct VFFVA results recorded where executable
[ ] benchmark limitations stated without substituting weaker evidence
[ ] docs attribute VFFVA correctly
[ ] no new runtime CPLEX/GLPK/MPI dependency
[ ] focused tests green
[ ] full relevant CI green
[ ] branch clean, commits pushed, PR opened
```

The lead may not declare completion with unchecked hard items unless the guardian documents a genuine external blocker and explains why it does not invalidate the implementation itself.

## 4.4 Token discipline

The guardian must also prevent token waste.

- Stop agents that begin broad repo-wide redesigns.
- Require agents to inspect only the files needed for their assigned responsibility.
- If an agent has tried the same failing approach twice without new evidence, require it to report the blocker and change strategy rather than continue meandering.
- Do not spawn duplicate agents to solve the same issue unless the guardian explicitly requests an independent audit.
- Prefer concrete code/test evidence over long speculative reports.

---

# 5. Mandatory specialist agents

After VFFVA-GUARDIAN exists, create the following specialists with non-overlapping primary responsibilities.

## Agent A — upstream VFFVA forensic translator

Responsibilities:

- audit upstream VFFVA at the pinned commit;
- extract the exact shared-memory execution sequence;
- map every relevant CPLEX operation to the closest HiGHS/highspy operation;
- distinguish essential VFFVA mechanics from CPLEX-specific or MPI-specific details;
- identify the exact semantics of `OMP_SCHEDULE`, default `dynamic,50`, max/min passes, solver clones, and per-thread solver configuration;
- produce a concise operation mapping and a list of behaviours the implementation must preserve.

This agent should not write production code until the guardian approves its mapping.

Required mapping should include at least:

```text
CPXopenCPLEX          -> worker-local Highs instance
CPXcloneprob          -> worker-local persistent model constructed from same prepared LP
CPX_PARAM_THREADS=1   -> Highs threads=1
CPXchgobjsen          -> setMaximize/setMinimize
CPXchgobj             -> changeColCost
CPXlpopt              -> Highs.run
CPXgetobjval           -> getObjectiveValue
OpenMP dynamic chunks -> shared-memory dynamic reaction-chunk queue
```

## Agent B — HiGHS concurrency and worker-lifecycle specialist

Responsibilities:

- inspect the pinned highspy/HiGHS bindings actually compatible with FluxEMU;
- verify that `Highs.run()` releases the Python GIL in the binding used by supported versions;
- investigate whether independent `Highs` instances are safe to run concurrently in Python threads when each is configured with `threads=1`;
- inspect relevant HiGHS global scheduler/threading caveats;
- design and run a focused stress test with multiple independent HiGHS instances in threads;
- determine whether any `Highs.resetGlobalScheduler()` handling is required and justify it with evidence;
- recommend the simplest correct shared-memory implementation.

Do not assume either safety or unsafety. Establish it.

If the agent claims threads are unusable, it must provide a minimal reproducer or authoritative upstream evidence. The guardian must reject a process fallback without such evidence.

## Agent C — faithful HiGHS VFFVA implementation engineer

Responsibilities:

- implement the production shared-memory engine after Agents A/B are approved;
- preserve existing public API where sensible, especially `run_highs_vffva(...)`;
- use one persistent worker-local HiGHS object per worker;
- keep worker-local simplex state alive across endpoint solves;
- keep the same worker alive across both objective-sense passes;
- implement a max pass followed by a min pass;
- implement dynamic reaction-chunk assignment beginning from VFFVA's default chunk size 50;
- set each worker's HiGHS internal thread count to 1;
- change only endpoint objective coefficient and objective sense in the hot loop;
- clear the endpoint coefficient after the solve;
- assemble results back into exact canonical reaction order;
- preserve contextual error reporting.

The current multiprocessing implementation may be retained temporarily as a benchmark/control during development, but absent a reproduced threading blocker it must not remain the normal production path when the milestone is complete.

Do not rewrite unrelated LP conditioning, sampling, or model code.

## Agent D — hot-loop performance auditor

Responsibilities:

Audit the current FluxEMU endpoint loop against upstream VFFVA.

In particular measure the cost of:

- retrieving the full primal after every endpoint;
- full independent mass-balance validation after every endpoint;
- repeated biological-objective reconstruction;
- repeated retained-objective checks;
- any basis clearing/recovery path;
- Python object allocation in the endpoint loop.

The agent must not simply delete safeguards.

Instead classify checks as:

```text
A. required in production hot path
B. can be moved to optional audit/debug mode
C. can be validated once per FVA call
D. should remain in cold-reference/parity tests rather than every production endpoint
```

If expensive full-primal validation materially harms the faithful VFFVA hot loop, design a safe separation such as a production-fast path plus explicit audit mode/tests. Preserve the cold reference as the permanent independent oracle.

The guardian must approve any removal/movement of validation before it lands.

## Agent E — correctness and regression engineer

Responsibilities:

Build/extend tests proving that the new threaded port has the same FVA mathematics as the existing cold reference.

At minimum test:

- dedicated analytical/synthetic FVA fixture;
- serial worker count 1;
- worker counts >1;
- fraction of optimum 1.0;
- fraction below 1.0;
- maximization;
- minimization;
- multi-term biological objective;
- fixed reactions;
- blocked reactions;
- signed/reversible reactions;
- canonical ordering independent of worker completion order;
- repeated deterministic bounds;
- worker-local model reuse instrumentation;
- exactly one worker solver object per active worker;
- no endpoint LP rebuild;
- contextual solver failure including reaction and direction;
- current FBA/FVA/cold-reference parity tests;
- no regression of Stage 1 sampled-state or EMU integration tests caused by the FVA replacement.

Do not weaken tolerances to make threaded results pass.

## Agent F — benchmark and upstream-comparison engineer

Responsibilities:

Create reproducible benchmark evidence that separates three questions:

### Internal correctness/engineering benchmark

```text
cold native reference
current/reusable serial HiGHS
new faithful threaded VFFVA->HiGHS path
```

This answers whether model reuse and shared-memory scheduling improve FluxEMU.

### Scaling benchmark

For the new HiGHS port record where hardware permits:

```text
workers = 1, 2, 4, 8, ...
```

Record:

- model reaction count;
- balanced-metabolite count;
- number of endpoint LPs;
- worker count;
- chunk size;
- scheduling mode;
- warm-up policy;
- repeated wall times;
- median wall time;
- endpoint solves per second;
- speedup `T1/Tp`;
- parallel efficiency `(T1/Tp)/p`;
- Python version;
- highspy/HiGHS version;
- machine/CPU information where practical.

Use at least:

1. the small E. coli-core model to expose overhead;
2. at least one medium or large genome-scale model where parallel scheduling has a chance to amortize overhead;
3. preferably a model/workload with heterogeneous endpoint solve times.

Do not conclude that parallelism is ineffective from the 95-reaction model alone.

### Direct upstream VFFVA comparison

Make a serious attempt to benchmark the original upstream VFFVA implementation using the same model/LP and matching objective-retention semantics.

Preferred order:

1. VFFVA + CPLEX if a valid CPLEX environment/license is already available;
2. otherwise the upstream VFFVA GLPK backend if build/runtime dependencies are available;
3. if neither can be executed, record the exact blocker and the commands attempted.

Do not add those backends as FluxEMU production dependencies.

For direct comparison, use a model where original VFFVA's single-objective-reaction assumption matches the benchmark LP. Do not distort a multi-term FluxEMU objective merely to force VFFVA compatibility.

The direct benchmark should compare identical reaction sets, objective percentage/fraction, and hardware where possible, and verify output bounds before reporting timing.

The task may still complete if CPLEX/GLPK execution is externally blocked, but the benchmark report must say clearly that the direct upstream comparison was attempted and blocked. Do not substitute the cold-reference 2x number and describe it as VFFVA performance evidence.

## Agent G — attribution/docs/API reviewer

Responsibilities:

- ensure the implementation and docs describe the method accurately as a HiGHS-native port/adaptation of VFFVA's computational architecture;
- cite Ben Guebila / VFFVA appropriately in relevant docs;
- ensure no text claims FluxEMU invented dynamic FastFVA scheduling;
- ensure no new CPLEX/GLPK/MPI runtime dependency appears;
- check public API and defaults;
- update stale docs that describe the multiprocessing design as final production architecture;
- keep documentation narrowly scoped to this port.

---

# 6. Required execution sequence

Do not let all agents code at once.

Use this sequence.

## Phase 1 — forensic design gate

1. Spawn VFFVA-GUARDIAN first.
2. Spawn Agents A and B.
3. Agents A/B report concise evidence and proposed mapping.
4. Guardian reviews and returns APPROVE/REJECT.
5. Lead writes the agreed implementation plan in its working notes.

No production implementation before this gate passes.

## Phase 2 — faithful engine implementation

1. Spawn Agent C after Phase 1 approval.
2. Agent C implements the smallest coherent shared-memory VFFVA->HiGHS core.
3. Agent E builds focused correctness/architecture tests in parallel, without redesigning C's implementation.
4. Guardian reviews implementation and tests.
5. Fix rejected items.
6. Run focused tests.
7. Commit and push the coherent engine milestone immediately.

Suggested commit theme:

```text
Port VFFVA shared-memory FVA architecture to HiGHS
```

## Phase 3 — hot-loop audit and performance refinement

1. Spawn Agent D against the committed engine.
2. Profile current hot-loop costs.
3. Make only evidence-backed changes that move nonessential audit work out of the production hot loop while preserving correctness through explicit audit/cold-reference tests.
4. Guardian approves every validation-contract change.
5. Run focused tests and benchmarks.
6. Commit and push.

Suggested commit theme:

```text
Align HiGHS FVA hot loop with VFFVA reuse semantics
```

## Phase 4 — scaling and original-VFFVA benchmark

1. Spawn Agent F.
2. Benchmark small + medium/large model(s).
3. Attempt direct original-VFFVA execution.
4. Verify output parity before timing claims.
5. Record reproducible benchmark evidence.
6. Do not encode brittle timing thresholds into normal CI.
7. Guardian audits the evidence and wording.
8. Commit benchmark/docs evidence if repository conventions support checked benchmark results; otherwise document exact reproducible commands.

Suggested commit theme:

```text
Benchmark HiGHS VFFVA port and scaling
```

## Phase 5 — docs/portability/final audit

1. Agent G updates only relevant docs/API descriptions.
2. Agent E runs final regression selection.
3. Guardian performs adversarial final review.
4. Fix every guardian REJECT item.
5. Run all relevant CI/workflows.
6. Open PR against `main`.
7. Stop.

---

# 7. Required technical architecture

The intended default shared-memory shape is conceptually:

```text
PreparedFluxRegion
        |
        +-- FBA already solved once
        +-- retained biological objective already imposed
        |
        v
Worker pool (threads)
        |
        +-- worker 0 owns Highs instance 0
        +-- worker 1 owns Highs instance 1
        +-- ...
        |
        |  every Highs instance:
        |      solver = simplex
        |      threads = 1
        |      LP/model built once
        |
        +--> MAX PASS
        |       shared dynamic queue of reaction chunks
        |       worker pulls next chunk when free
        |       per reaction:
        |           change endpoint cost -> +1
        |           maximize
        |           Highs.run()
        |           read objective/status
        |           clear endpoint cost
        |
        +--> MIN PASS
                same workers, same Highs objects
                shared dynamic queue of reaction chunks
                per reaction:
                    change endpoint cost -> +1
                    minimize
                    Highs.run()
                    read objective/status
                    clear endpoint cost

canonical result assembly
```

The exact Python concurrency primitive is up to the implementation, but it must be genuinely shared-memory and must allow each thread to retain ownership of one persistent HiGHS instance.

A generic executor pattern that can arbitrarily move endpoint tasks between threads without preserving worker-local solver ownership is not sufficient unless the implementation explicitly supplies thread-local worker state that guarantees one persistent solver per thread.

A simple explicit worker-thread + shared queue design is acceptable and may be clearer.

Start from VFFVA's `dynamic,50` scheduling behaviour rather than inventing a new default.

If measurement shows another chunk size is consistently better for HiGHS, keep 50 as the faithful baseline in benchmark evidence and justify any changed production default empirically. Do not silently tune the implementation before recording the faithful baseline.

---

# 8. Correctness acceptance

The cold native reference remains the permanent mathematical oracle.

For every tested model/case, require:

```text
same canonical reaction order
same min/max endpoint set
same retained biological objective semantics
max absolute endpoint discrepancy <= existing accepted FVA tolerance
```

Use existing repository tolerances unless there is a documented numerical reason to do otherwise. Do not relax them for convenience.

At minimum compare new fast results against the cold reference on all cases already covered by native FVA tests and all newly relevant concurrency cases.

Where COBRApy parity tests already exist, keep them as an optional independent compatibility oracle; do not add COBRApy to the production path.

For a direct original-VFFVA benchmark, verify returned min/max bounds first. Timing numbers without semantic parity are invalid evidence.

---

# 9. Architecture acceptance instrumentation

Tests/instrumentation must be able to demonstrate, not merely assert, that:

- model compilation occurs once per prepared analysis where expected;
- biological FBA solve occurs once per prepared/composed analysis where expected;
- worker count `p` creates at most `p` persistent HiGHS solver instances;
- each active worker reuses its solver across many endpoints;
- solver instances survive from max pass into min pass;
- HiGHS internal `threads` is 1 in worker-level parallel mode;
- endpoint objective changes occur without rebuilding the LP;
- dynamic work distribution actually allows a worker to take additional chunks after completing earlier work;
- final result order does not depend on completion order.

Do not fake these tests with comments or counters disconnected from actual worker lifecycle.

---

# 10. Performance acceptance

Performance is evidence, not a brittle CI gate.

However, the new implementation must not be declared successful merely because it is correct.

Required evidence:

1. serial reusable HiGHS remains materially faster than cold rebuild FVA on a representative model;
2. threaded VFFVA-style execution avoids the extreme process-spawn/IPC overhead seen in the previous multiprocessing path on small models;
3. scaling behaviour is measured on at least one model large enough to make multiple workers meaningful;
4. direct upstream VFFVA comparison is attempted and recorded;
5. all benchmark results include enough environment metadata to reproduce/interpret them.

Do not require FluxEMU+HiGHS to beat VFFVA+CPLEX. That is not the scientific contribution and different solvers may have different performance.

The benchmark question is:

> Does the HiGHS translation faithfully reproduce VFFVA's reusable-LP/dynamic-load-balancing architecture, and what performance does that architecture deliver with HiGHS?

Report the answer, whatever it is.

---

# 11. What to do with the current multiprocessing implementation

The existing multiprocessing/spawn implementation is useful as historical evidence and perhaps a temporary benchmark control.

It is not the desired final production architecture for this task.

Allowed outcomes:

### Preferred

- replace it internally with the new shared-memory implementation;
- preserve `run_highs_vffva(...)` public semantics;
- delete obsolete private process-worker machinery after tests prove it is unnecessary.

### Acceptable temporary compatibility outcome

- retain the old process engine only as a private benchmark/reference helper or explicitly named fallback;
- production default uses the new threaded implementation.

### Not acceptable without guardian-approved reproduced blocker

- leave multiprocessing as production and merely tune chunk size/work ordering;
- claim that is the VFFVA-to-HiGHS port.

---

# 12. Failure handling

Preserve clear failures.

A failed endpoint should identify at least:

```text
reaction id
min/max direction
solver status/message
```

If a worker thread fails, ensure the overall FVA call terminates cleanly and does not deadlock while other workers wait on a queue/barrier.

Add tests for worker failure propagation and shutdown.

Do not silently drop endpoint failures or return partial FVA tables.

---

# 13. CI and regression scope

At minimum run the focused suites touching:

```text
native HiGHS FBA/FVA
native VFFVA/FastFVA
COBRA parity where available
Stage 1 ensemble integration
native stationary analysis
clean-install/native portability
real-model acceptance where practical
```

The exact repository test commands should follow current docs/CI.

Do not modify unrelated tests just to reduce runtime.

A benchmark timing ratio should not become a hard GitHub Actions pass/fail criterion because hosted-runner timing is noisy.

Architecture and numerical correctness must be hard gates.

---

# 14. Documentation requirements

Update relevant documentation to make these points explicit:

1. FluxEMU uses a HiGHS-native translation/adaptation of VFFVA's reusable-LP and dynamic-load-balancing strategy.
2. VFFVA/Ben Guebila is credited for that computational architecture.
3. FluxEMU preserves more general linear-objective/retention semantics than the original VFFVA implementation in places where upstream assumes a single objective reaction.
4. production uses HiGHS and does not require CPLEX, GLPK, MPI, or the original VFFVA binary.
5. the previous multiprocessing implementation is no longer the default architecture.
6. benchmark evidence clearly distinguishes:
   - cold-reference speedup;
   - threaded scaling;
   - direct original-VFFVA comparison.

Do not overclaim novelty or performance.

---

# 15. Commit discipline

Obey `AGENTS.md`.

At a minimum produce coherent pushed commits corresponding to:

1. shared-memory faithful engine + core tests;
2. hot-loop validation/performance alignment if changes are needed;
3. benchmark/direct-VFFVA evidence + docs/CI finalization.

Do not leave substantial work only in the final working tree.

Do not squash all engineering into one opaque final commit during execution.

---

# 16. Final VFFVA-GUARDIAN adversarial audit

Before opening the PR, the guardian must inspect the final diff and answer each of these explicitly.

### Fidelity

- Is this genuinely a shared-memory VFFVA-style architecture, or did the team merely optimize the old multiprocessing implementation?
- Does each worker own one persistent HiGHS model?
- Are the same workers/models reused across max and min passes?
- Is dynamic chunk scheduling actually implemented?
- Did the implementation begin from VFFVA's `dynamic,50` baseline?
- Are endpoint LPs reoptimised rather than rebuilt?

### Scientific correctness

- Are FluxEMU's multi-term/minimization/retained-objective semantics preserved?
- Did any code assume a single objective reaction merely because VFFVA does?
- Are canonical order and endpoint parity preserved?
- Were tolerances weakened?

### Performance honesty

- Was the small E. coli-core overhead regime distinguished from useful scaling regimes?
- Was at least one medium/large model benchmarked?
- Was original VFFVA execution actually attempted?
- If original VFFVA could not run, is the blocker concrete and documented?
- Is the old cold-reference speedup clearly labelled as internal engineering evidence rather than external VFFVA comparison?

### Scope discipline

- Did anyone modify sampling, MFA, EMU mathematics, atom mappings, or information-theory code unnecessarily?
- Were unrelated cleanups kept out?
- Did new runtime dependencies creep in?

### Repository state

- focused tests pass;
- relevant CI passes;
- no required TODOs remain;
- commits are pushed;
- working tree is clean;
- PR is open against `main`.

The guardian must return:

```text
FINAL APPROVE
```

or

```text
FINAL REJECT
<concrete list of required fixes>
```

The lead agent is forbidden from opening the final completion PR or reporting the task complete on `FINAL REJECT`.

---

# 17. Required final report

When the task is complete, provide a concise report containing:

1. branch name and final commit SHA;
2. PR number/link;
3. exact mapping from VFFVA architecture to HiGHS implementation;
4. whether production multiprocessing was removed, retained only as fallback/reference, or otherwise handled;
5. worker lifecycle and dynamic scheduling design;
6. validation moved out of the hot loop, if any, and where correctness is now enforced;
7. test commands/results;
8. benchmark models and environment;
9. worker scaling table/summary;
10. direct original-VFFVA benchmark result, or exact blocker if it could not be run;
11. attribution/docs changes;
12. VFFVA-GUARDIAN final verdict.

Do not continue into another milestone after this report.
