"""An evaluation-domain change must not masquerade as improved predictions."""
import copy

import numpy as np
import pytest

from scripts.audit_native_body_domains import matched_domains
from scripts.audit_native_body_reference import metrics


def panels():
    old = {k: np.zeros((1, 4, 4), bool) for k in ('positive', 'ordinary', 'foreign')}
    old['pixels'] = np.full((1, 4, 4), 128, np.uint8)
    old['positive'][0, 1, 1] = True
    old['ordinary'][0, 1, 2:] = True
    old['foreign'][0, 3, 3] = True
    new = {k:v.copy() for k,v in old.items()}
    new['ordinary'][0, 1, 3] = False
    doc = {'cases':[{'id':'a', 'grain':[1,1]}],
           'supervision_contract':'finalized_self_reviewed_background_foreign_v1'}
    return old, new, doc


def test_same_false_prediction_changes_only_the_domain_score():
    old, new, doc = panels()
    domains = matched_domains(old, doc, new, doc)
    prediction = np.zeros((4,4))
    prediction[1,1] = prediction[1,3] = .9
    values = {name:metrics(prediction, *[domain[k][0] for k in ('positive','ordinary','foreign')])
              for name,domain in domains.items()}
    assert values['legacy']['iou'] == .5
    assert values['corrected']['iou'] == values['common_reviewed']['iou'] == 1.
    assert all(row['recall'] == 1. for row in values.values())
    assert prediction[1,3] == .9  # No change in the model's incorrect legacy-domain prediction.


@pytest.mark.parametrize('change', ['pixels','positive','foreign','ordinary','grain'])
def test_domain_comparison_refuses_changed_images_truth_or_queries(change):
    old, new, doc = panels()
    revised = copy.deepcopy(doc)
    if change == 'grain':
        revised['cases'][0]['grain'] = [2,2]
    else:
        new[change][0,0,0] = 1
    with pytest.raises(ValueError):
        matched_domains(old,doc,new,revised)
