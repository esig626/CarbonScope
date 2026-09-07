# Bruno v2 theorem transfer

FluxEMU implements the order-specific finite-sample Type-II lower bound used in the simple-binary testing layer from Bruno, Vandenbroucque & Esposito, arXiv:2601.09550v2.

The implementation uses the fixed roles

```text
H0=P0=null, H1=P1=alternative,
Type I=P0(decide H1), Type II=P1(decide H0).
```

The reverse component uses `D_lambda(P1 || P0)` and the forward component uses `D_lambda(P0 || P1)`. Both are evaluated on the complete observation laws. For genuine multinomial blocks, FluxEMU reuses the separately tested identity `D_alpha(Mult(n,p)||Mult(n,q)) = n D_alpha(p||q)` and independent-product additivity.

The implementation requires exact mutual absolute continuity and retains structural zeros. It neither smooths distributions nor substitutes a finite Rényi-order grid for the source theorem's continuous family of admissible real orders above one.

Independent validation in `tests/test_testing_oracles.py` constructs complete finite count laws with exact rational arithmetic, evaluates Rényi divergences independently in high-precision decimal arithmetic, and exhausts deterministic rejection regions on bounded test spaces. This validation is intentionally independent of the production Bruno routine.

`bruno_converse_at_order(...)` returns `BrunoOrderBound`, an order-specific lower bound. It does not state that the lower bound is achieved and makes no claim that a global continuous-order envelope has been evaluated.
