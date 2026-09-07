# Source policy

No Version-1 map is accepted from an automatic atom mapper alone. The
`provenance` object is mandatory and records source type, full citation,
source identifier, exact source location, and curation notes.

The precedence is:

1. Published isotope-MFA/EMU models with explicit transitions.
2. Established manually curated atom-transition resources.
3. Primary biochemical structural literature.
4. Automatic mapping only as a diagnostic, never as validation authority.

The Antoniewicz Table 5 entries are `gold`: their normalized form is checked
against the frozen `TABLE5_REACTIONS` transcription and the published Table 6
MID. Other entries derived from explicit maps in the official mfapy examples
are `curated`. `provisional` is reserved for a source-audited but not yet
fully validated record; Version 1 does not use a guessed map merely to fill a
coverage gap.

The source model can contain a net reaction. In that case the library calls it
a net transition and does not pretend it validates unexposed intermediate
steps. The oxidative PPP G6P-to-6PG net record is the key example.
