# Antoniewicz TCA independent validation

This directory retains a small independent validation of the stationary native EMU engine against the eight reaction TCA example in Antoniewicz, Kelleher and Stephanopoulos (2007).

* `direct_isotopomer_solver.py` enumerates the full isotopomer state and solves the published balances directly with NumPy. It does not call CarbonScope's EMU implementation.
* `published_reference_mid.csv` records the published Table 6 glutamate MID used as a rounded literature reference.
* `PAPER_TRANSCRIPTION.md` records the source transcription.

The executable regression is `tests/test_antoniewicz_native.py`. Legacy backend and model builder machinery is intentionally not part of this validation.
