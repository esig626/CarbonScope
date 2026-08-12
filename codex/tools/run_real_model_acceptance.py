"""Final CI programme acceptance: flux analysis, positive EMU fixtures, negative oracle."""
from __future__ import annotations
from dataclasses import asdict
import importlib.resources, json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
from fluxemu.analysis import run_native_fba, run_native_fva
from fluxemu.emu import compile_emu_plan, evaluate_stationary
from fluxemu.emu.projection import evaluate_flux_projection
from fluxemu.exceptions import ForwardEMUError
from fluxemu.execution import CanonicalFluxState
from fluxemu.real_model import build_r1_acceptance_experiment, load_ecoli_core_stage_b2_model

ROOT=Path('codex/results/real_model_50_50_u13c_glucose/final_native_highs')
EXPECTED={'biomass':0.8739215069684307,'acetate':12.030622385803783}; TOL=1e-7
NAMES=('01_pyruvate.png','02_alanine.png','03_lactate.png','04_citrate.png','05_akg.png','06_succinate.png','07_fumarate.png','08_malate.png','09_glutamine.png','10_aspartate.png','11_glycine.png','12_serine.png')

def validate(condition,model,fba):
 if abs(fba.objective_value-EXPECTED[condition])>TOL: raise RuntimeError(f'{condition} objective parity failed')
 d=fba.diagnostics
 if max(d.max_mass_balance_residual,d.max_lower_bound_violation,d.max_upper_bound_violation)>TOL: raise RuntimeError(f'{condition} primal feasibility failed')
 if abs(fba.fluxes['EX_glc__D_e']+10)>TOL: raise RuntimeError(f'{condition} glucose changed')
 if fba.fluxes['BIOMASS_Ecoli_core_w_GAM']<0.5243529041810849-TOL: raise RuntimeError(f'{condition} viability failed')

def projected(model,flux):
 rules={x.flux_projection.projection_id:x.flux_projection for x in model.isotope_model.reactions}
 return {i:evaluate_flux_projection(r,flux) for i,r in rules.items()}

def write_mid(out,result,meta):
 out.mkdir(parents=True,exist_ok=True); preds=result.forward.predictions
 pd.DataFrame({'target':q.target_id,'isotopologue':f'M+{i}','fraction':v} for q in preds for i,v in enumerate(q.fractions)).to_csv(out/'mids.csv',index=False)
 plots=out/'plots';plots.mkdir(exist_ok=True)
 for q,name in zip(preds,NAMES,strict=True):
  fig,ax=plt.subplots(figsize=(5,3.2));ax.bar([f'M+{i}' for i in range(len(q.fractions))],q.fractions);ax.set_ylim(0,1);ax.set_ylabel('fractional abundance');ax.set_title(q.target_id);fig.tight_layout();fig.savefig(plots/name,dpi=150);plt.close(fig)
 meta.update({'maximum_mid_normalization_error':result.forward.max_normalization_error,'layers':[asdict(x) for x in result.layer_diagnostics]});(out/'diagnostics.json').write_text(json.dumps(meta,indent=2)+'\n')

def load_fixtures():
 p=importlib.resources.files('fluxemu.real_model')/'data/e_coli_core_positive_fixtures.json';return json.loads(p.read_text())['selection']

def run():
 experiment=build_r1_acceptance_experiment(); model=load_ecoli_core_stage_b2_model(); plan=compile_emu_plan(model,experiment)
 # A: native programme correctness, independent of isotope suitability.
 for condition in ('biomass','acetate'):
  m=load_ecoli_core_stage_b2_model(condition);a=run_native_fba(m);b=run_native_fba(m);fva=run_native_fva(m,1.0)
  validate(condition,m,a)
  if not a.fluxes.equals(b.fluxes) or a.objective_value!=b.objective_value:raise RuntimeError(f'{condition} native FBA nondeterministic')
  out=ROOT/'flux_analysis'/condition;out.mkdir(parents=True,exist_ok=True);a.fluxes.rename('flux').to_csv(out/'fba_fluxes.csv',index_label='reaction_id');fva.ranges.to_csv(out/'fva_ranges.csv',index_label='reaction_id');pd.Series(projected(m,a.fluxes.to_dict()),name='rate').to_csv(out/'projected_isotope_fluxes.csv',index_label='projection_id');(out/'diagnostics.json').write_text(json.dumps({'objective':a.objective_value,'fba':asdict(a.diagnostics),'deterministic_repeat':True},indent=2)+'\n')
 # B/C: authoritative complete positive fixtures.
 fixtures=load_fixtures()
 for condition in ('biomass','acetate'):
  x=fixtures[condition];state=CanonicalFluxState(x['sample_id'],tuple(map(tuple,x['fluxes'])));first=evaluate_stationary(plan,(state,));second=evaluate_stationary(plan,(state,))
  if first!=second or len(first.forward.predictions)!=12:raise RuntimeError(f'{condition} positive EMU fixture failed')
  out=ROOT/'positive_emu'/condition;out.mkdir(parents=True,exist_ok=True);pd.Series(projected(model,dict(state.values)),name='rate').to_csv(out/'projected_isotope_fluxes.csv',index_label='projection_id');pd.Series(dict(state.values),name='flux').to_csv(out/'fluxes.csv',index_label='reaction_id');write_mid(out,first,{'sample_id':x['sample_id'],'source':x['source'],'reason_selected':x['reason_selected'],'deterministic_repeat':True,'noise':'none','sampling':'none','inverse_mfa':False})
 # D: exact historical acetate optimum must fail cleanly, never be regularised.
 frozen=json.loads((importlib.resources.files('fluxemu.real_model')/'data/e_coli_core_frozen_optima.json').read_text());x=frozen['acetate'];state=CanonicalFluxState(x['sample_id'],tuple(map(tuple,x['fluxes'])));out=ROOT/'negative_oracles'/'acetate_exact_optimum';out.mkdir(parents=True,exist_ok=True);pd.Series(dict(state.values),name='flux').to_csv(out/'fluxes.csv',index_label='reaction_id')
 try:evaluate_stationary(plan,(state,));raise RuntimeError('acetate exact optimum unexpectedly defined all MIDs')
 except ForwardEMUError as e:
  message=str(e)
  if not all(s in message for s in ('layer 4','dimension=4','rank=1','oaa_c','mal__L_c','fum_c','succ_c')):raise
  (out/'diagnostics.json').write_text(json.dumps({'status':'expected_negative_pass','error':message,'flux_unchanged':True,'regularisation':False,'mids_fabricated':False},indent=2)+'\n')
 if len(tuple((ROOT/'positive_emu').glob('*/plots/*.png')))!=24:raise RuntimeError('positive programme fixtures did not create exactly 24 PNGs')
 verdict='FLUXEMU REAL-MODEL PROGRAMME ACCEPTANCE — PASSED';(ROOT/'VERDICT.txt').write_text(verdict+'\n');print(verdict);return 0
if __name__=='__main__':sys.exit(run())
