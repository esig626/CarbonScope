"""Non-gating JSON benchmark for COBRA and cold native reference FVA."""
from __future__ import annotations
import argparse, json, statistics, time
from cobra.io import read_sbml_model
from fluxemu.cobra_analysis import run_fba, run_fva
from fluxemu.compat import project_cobra_flux_model
from fluxemu.flux_analysis import compile_flux_lp, run_highs_fba, run_highs_fva_reference

def median_time(call, repeats):
    values=[]
    for _ in range(repeats):
        start=time.perf_counter(); call(); values.append(time.perf_counter()-start)
    return statistics.median(values)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("--repeats", type=int, default=3)
    args=parser.parse_args(); cobra=read_sbml_model(args.model); canonical=project_cobra_flux_model(cobra)
    run_fva(cobra, 1.0); run_highs_fva_reference(canonical, 1.0)  # warm-up
    reactions=len(canonical.reactions); repeats=args.repeats
    data={"model":args.model,"reactions":reactions,"repeats":repeats,"diagnostic_only":True,
          "compile_median_seconds":median_time(lambda:compile_flux_lp(canonical), repeats),
          "native_fba_median_seconds":median_time(lambda:run_highs_fba(canonical), repeats),
          "native_cold_fva_median_seconds":median_time(lambda:run_highs_fva_reference(canonical,1), repeats),
          "cobra_fva_processes_1_median_seconds":median_time(lambda:run_fva(cobra,1.0), repeats),
          "native_endpoint_lp_solves":2*reactions,
          "native_seconds_per_endpoint":None}
    data["native_seconds_per_endpoint"]=data["native_cold_fva_median_seconds"]/(2*reactions)
    print(json.dumps(data, sort_keys=True))
if __name__ == "__main__": main()
