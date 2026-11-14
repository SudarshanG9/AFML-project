"""Independent sign/undefined-value tests, not a replacement evaluator."""
from copy import deepcopy
import pytest
from avfuse.member1_fusion.dayc_research import build_h2_h3_decision_table,describe_limitations,interval


def evidence():
    return {'status':'SYNTHETIC','losses':{'reference':0.5,'low_precision':0.6,'restore_value_output':0.52,'restore_query':0.55},
            'damage_ci':[0.01,0.2],'recovery_ci':{'restore_value_output':[0.01,0.12],'restore_query':[-0.02,0.09]},
            'h3_ci':[-0.01,0.07],'costs':{'E_value_ms':2.0,'E_query_ms':1.0},'cost_reliability_approved':True}


def test_correct_h3_sign_and_no_indirect_significance():
    table=build_h2_h3_decision_table(evidence())
    assert table['H3_direct']==pytest.approx(0.03)
    assert not table['value_better_than_query_supported']
    assert table['H3_verdict'].startswith('inconclusive')
    assert table['recoveries']['restore_value_output']['positive_recovery_supported']
    assert not table['recoveries']['restore_query']['positive_recovery_supported']
    assert table['Rv_per_ms']==pytest.approx(0.04)
    assert not table['equal_cost_established'] and not table['policy_selected']


@pytest.mark.parametrize('damage',[0.,-0.1])
def test_nonpositive_damage_disables_share_and_rescue(damage):
    e=evidence();e['losses']['reference']=e['losses']['low_precision']-damage
    table=build_h2_h3_decision_table(e)
    for r in table['recoveries'].values():
        assert r['share_recovered'] is None and not r['positive_recovery_supported']
    assert 'unsubstantiated' in table['damage_interpretation']


def test_damage_ci_spanning_zero_does_not_support_harm():
    e=evidence();e['damage_ci']=[-0.1,0.2]
    table=build_h2_h3_decision_table(e)
    assert not table['positive_damage_supported']
    assert not table['recoveries']['restore_value_output']['positive_recovery_supported']


@pytest.mark.parametrize('cost',[0.,-1.,None,float('nan')])
def test_unavailable_or_nonpositive_cost_is_undefined(cost):
    e=evidence();e['costs']['E_query_ms']=cost
    t=build_h2_h3_decision_table(e)
    assert t['Rv_per_ms'] is t['Rq_per_ms'] is None


def test_unreviewed_timing_is_not_reliable():
    e=evidence();e['cost_reliability_approved']=False
    t=build_h2_h3_decision_table(e)
    assert t['Rv_per_ms'] is None and t['Rq_per_ms'] is None


@pytest.mark.parametrize('ci',[{'ci_low':float('nan'),'ci_high':1},{'ci_low':2,'ci_high':1}])
def test_invalid_interval_fails(ci):
    with pytest.raises(ValueError):interval(ci)


def test_mandatory_limitations():
    result=' '.join(describe_limitations({'blockers':['owner handoff missing']}))
    for word in ('Silence','test-label','Fake INT8','cumulative','timing','owner handoff'):assert word in result


def test_day9_decision_requires_actual_complete_packet_and_never_selects():
    from avfuse.member1_fusion.dayc_research import validate_day9_decision
    e=evidence();e['status']='RESEARCH_REVIEW_REQUIRED';e['blockers']=[]
    e['costs']['status']='PASS'
    e['per_type_results']={name:{'n':1200 if name=='overall' else 600,'contrasts':[
        {'a':'restore_value_output','b':'restore_query','metrics':{'bce':{'delta':0.03,'ci_low':-0.01,'ci_high':0.07}}}]}
        for name in ('overall','audio_fake','video_fake','both_fake')}
    packet={'status':'RESEARCH_REVIEW_REQUIRED','policy_frozen':False,'evidence':e,'decision_table':build_h2_h3_decision_table(e)}
    result=validate_day9_decision(packet)
    assert result['H2']=='supported' and result['H3']=='inconclusive' and not result['policy_selected']
    packet['decision_table']['H3_direct']=-9
    with pytest.raises(ValueError):validate_day9_decision(packet)
