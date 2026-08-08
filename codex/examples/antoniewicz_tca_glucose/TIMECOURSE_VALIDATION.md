# Time-resolved mfapy `diffmdv` validation

The existing mfapy `diffmdv` route is used through the model's generated
function. All non-source/non-excreted intracellular pools (including the
terminal glutamate EMUs) start at M+0 and use 100 arbitrary pool units. Flux
units and pool units are arbitrary, so time is numerical rather than
biological. The fixed flux vector is never altered to accelerate convergence.

The requested initial time grid was `0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10`.
The final point was not converged, so the script automatically extended it to
`20, 40, 80, 160`. At time 160 the largest absolute difference from the
stationary `calmdv` result across every target and MID component was
`2.81454249e-7`, below the declared `1e-5` convergence criterion.

Vectors in the table are respectively citrate M+0..M+6; OAC M+0..M+4; and
glutamate M+0..M+5. They are the values saved in `timecourse_mids.csv`.

| Time | citrate; OAC; glutamate |
| ---: | --- |
| 0 | `1, 0, 0, 0, 0, 0, 0; 1, 0, 0, 0, 0; 1, 0, 0, 0, 0, 0` |
| 0.05 | `1, 0, 0, 0, 0, 0, 0; 1, 0, 0, 0, 0; 1, 0, 0, 0, 0, 0` |
| 0.1 | `1, 0, 0, 0, 0, 0, 0; 1, 0, 0, 0, 0; 1, 0, 0, 0, 0, 0` |
| 0.25 | `1, 0, 0, 0, 0, 0, 0; 1, 0, 0, 0, 0; 1, 0, 0, 0, 0, 0` |
| 0.5 | `1, 0, 0, 0, 0, 0, 0; 1, 0, 0, 0, 0; 1, 0, 0, 0, 0, 0` |
| 1 | `1, 0, 3.808964e-14, 0, 0, 0, 0; 1, 0, 0, 0, 0; 1, 0, 0, 0, 0, 0` |
| 2 | `0.99999995, 0, 5.1059937e-08, 0, 6.7460344e-25, 0, 0; 1, 0, 1.0170994e-15, 0, 0; 1, 0, 2.2199803e-10, 0, 0, 0` |
| 5 | `0.9995185, 4.4376064e-11, 0.00048150298, 7.5866752e-14, 4.3845765e-10, 1.8840956e-17, 1.573367e-29; 0.99999849, 2.8990573e-10, 1.5107459e-06, 5.083697e-14, 9.6806457e-24; 0.99996664, 6.6115273e-09, 3.3356022e-05, 3.0133073e-12, 3.0130815e-12, 6.9717793e-21` |
| 10 | `0.93378926, 6.3223548e-06, 0.066107808, 7.4679638e-07, 9.58443e-05, 1.6133277e-08, 1.5935053e-13; 0.9976627, 1.6457627e-05, 0.0023202451, 6.0201338e-07, 7.906572e-12; 0.98334252, 8.6411333e-05, 0.016561533, 4.7846556e-06, 4.7531978e-06, 1.5007129e-10` |
| 20 | `0.21626493, 0.001820047, 0.65756337, 0.0074974252, 0.11103638, 0.0057692836, 4.8566134e-05; 0.80338549, 0.0095355078, 0.17561242, 0.011351275, 0.00011530681; 0.43005878, 0.014947315, 0.50297498, 0.026147374, 0.025487709, 0.00038384127` |
| 40 | `0.00024775167, 1.2298667e-06, 0.50296765, 0.0016240734, 0.26431978, 0.18192193, 0.048917587; 0.50211134, 0.0011109486, 0.26133803, 0.18260607, 0.052833598; 0.0012847791, 0.00033306173, 0.51051287, 0.14076589, 0.267366, 0.079737399` |
| 80 | `0, 0, 0.50000234, 2.3748888e-06, 0.25000512, 0.1667416, 0.083248568; 0.50000131, 1.6878991e-06, 0.25000644, 0.16672666, 0.08326389; 1.1485118e-07, 2.2174092e-07, 0.50001146, 0.12499395, 0.25012173, 0.12487253` |
| 160 | `0, 1.2495441e-13, 0.5, 0, 0.25, 0.16666647, 0.083333527; 0.5, 0, 0.25, 0.16666651, 0.083333492; 0, 0, 0.49999999, 0.12500001, 0.24999972, 0.12500028` |

At the first useful citrate-label time (time 2), citrate M+2 is
`5.1059937e-08` while M+4 is `6.7460344e-25`; thus first labelled acetyl-CoA
enters initially unlabelled OAC as M+2. Citrate M+4 is then strictly positive
at time 5 (`4.3845765e-10`) and plainly visible by time 10
(`9.58443e-05`), when returned labelled OAC combines with labelled AcCoA.
Glutamate M+3 through M+5 also emerge later (their time-10 sum is about
`9.538e-06`). All written probability vectors are finite, nonnegative, and
normalized; sub-microfraction negative `odeint` roundoff is clipped to zero
and the vector renormalized before output.
