from pathlib import Path
import json
import pytest
from fluxemu.model import load_sbml_flux_model
from fluxemu.pipeline import run_pipeline
from fluxemu.native_io import load_native_stationary_spec
from fluxemu.exceptions import MappingError

FIXTURE=Path(__file__).parent/'fixtures'/'native_portability'
def test_sbml_native_end_to_end(tmp_path):
    flux=load_sbml_flux_model(FIXTURE/'model.xml')
    assert [r.reaction_id for r in flux.reactions] == ['foreign_hx','foreign_sink']
    result=run_pipeline(FIXTURE/'model.xml',FIXTURE/'experiment.yaml',tmp_path)
    assert result.analysis.fba.objective_value == pytest.approx(10)
    assert tuple(result.analysis.fba.fluxes) == pytest.approx((10,10))
    assert result.analysis.fva.ranges.loc['foreign_hx','minimum'] == pytest.approx(10)
    mids=json.loads((tmp_path/'predicted_mids.json').read_text())
    assert [x['predicted_fraction'] for x in mids] == pytest.approx([.5,0,0,0,0,0,.5])
    assert result.manifest['sampling_performed'] is False

def test_required_mapping_fails_explicitly(tmp_path):
    text=(FIXTURE/'experiment.yaml').read_text().replace('  assignments:\n','  assignments: []\n  ignored:\n')
    path=tmp_path/'bad.yaml'; path.write_text(text)
    with pytest.raises(MappingError,match='missing authoritative mapping'):
        load_native_stationary_spec(path,load_sbml_flux_model(FIXTURE/'model.xml'))
