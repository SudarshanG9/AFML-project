"""Compact, source-bound result presentation; no detector or evaluation calculations."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .data_access import CONFIGS, saved_result_table, load_json, documented_test_results
from .workbench_service import load_experiment_results, evaluate_validation_results


def performance_rows(configs):
    rows = []
    for mode in CONFIGS:
        if mode not in configs:
            continue
        block = configs[mode]
        row = {'Configuration': mode}
        for metric in ('auroc', 'auprc', 'recall', 'bce'):
            value = block.get(metric, {}).get('value')
            row[metric.upper()] = value if isinstance(value, (float, int)) and math.isfinite(value) else None
        rows.append(row)
    return rows


def _performance(configs):
    st.subheader('Overall performance')
    rows = performance_rows(configs)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={name: st.column_config.NumberColumn(format='%.6f') for name in ('AUROC', 'AUPRC', 'Recall', 'BCE', 'RECALL')})
        st.caption('Unavailable metrics remain empty. Confidence intervals and source provenance are in technical details.')
    else:
        st.info('No label-grounded performance metrics available for this source.')


def _manipulations(rows):
    st.subheader('Manipulation-type comparison')
    if not rows:
        st.info('Manipulation-stratified metrics are unavailable for this source.')
        return
    frame = pd.DataFrame(rows)
    chart = px.bar(frame, x='Manipulation', y='Recall', color='Configuration',
                   error_y='CI above', error_y_minus='CI below', barmode='group',
                   range_y=[0, 1.02], template='plotly_white',
                   color_discrete_sequence=['#6554c0', '#a69acb', '#3e938d', '#c49852'])
    chart.update_layout(height=320, margin=dict(l=0, r=0, t=12, b=0), legend_title_text='',
                        legend=dict(orientation='h', y=-.25), xaxis_title=None, yaxis_title='Recall at frozen threshold')
    st.plotly_chart(chart, use_container_width=True)
    st.caption('Evaluator strata: each fake type versus all real clips; recall intervals resample fake source groups within each type.')


def _strata(configs):
    rows = []
    for mode, block in configs.items():
        for kind, stats in block.get('per_type', {}).items():
            recall = stats.get('recall', {})
            value, lo, hi = (recall.get(name) for name in ('value', 'ci_low', 'ci_high'))
            if kind not in ('audio_fake', 'video_fake', 'both_fake') or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (value, lo, hi)):
                continue
            if not 0 <= lo <= value <= hi <= 1:
                continue
            rows.append({'Configuration': mode, 'Manipulation': kind.replace('_', ' '), 'Recall': value,
                         'CI above': hi-value, 'CI below': value-lo})
    return rows


def decision_comparison(rows, clip_id, threshold=None):
    selected = {row['config']: row['score_fake'] for row in rows if row['clip_id'] == clip_id}
    ref = selected.get('reference')
    result = []
    for mode in CONFIGS:
        score = selected.get(mode)
        item = {'Configuration': mode, 'Fake score': score,
                'Δ from FP32': score-ref if score is not None and ref is not None else None,
                'Availability': 'Available' if score is not None else 'Missing configuration'}
        if threshold is not None:
            item['Decision flip vs FP32'] = ((score >= threshold) != (ref >= threshold)) if score is not None and ref is not None else None
        result.append(item)
    return result


def _decisions(rows, threshold=None):
    st.subheader('Precision decision comparison')
    if not rows:
        st.info('No sample-aligned prediction artifacts available. Aggregate metrics cannot reconstruct per-clip scores.')
        return
    clip = st.selectbox('Inspect clip', sorted({row['clip_id'] for row in rows}), key='result_clip')
    st.dataframe(pd.DataFrame(decision_comparison(rows, clip, threshold)), hide_index=True, use_container_width=True)
    st.caption('Agreement with FP32 is not ground-truth correctness. Missing configurations remain missing.')
    if threshold is None:
        st.caption('Decision flips unavailable: no matching documented validation-fixed threshold report is associated with this run.')
    else:
        st.caption(f'Decisions use the associated frozen validation threshold: {threshold:.16g}.')


def _saved(root):
    from .ui import MODE_HELP
    split = st.radio('Accepted study split', ['test', 'val'], horizontal=True,
                     format_func=lambda s: 'Held-out aggregate evidence' if s == 'test' else 'Validation aggregate evidence')
    overall_name = 'test_overall_metrics.csv' if split == 'test' else 'validation_overall_metrics.csv'
    try:
        overall = saved_result_table(root, overall_name)
    except (OSError, ValueError) as exc:
        st.warning(f'Saved evidence unavailable: {exc}')
        return
    modes = st.session_state.get('experiment_draft', {}).get('modes', list(CONFIGS))
    configs = {}
    for row in overall:
        if row['mode'] not in modes: continue
        configs.setdefault(row['mode'], {'per_type': {}})[row['metric']] = {
            name: float(row[name]) for name in ('value', 'ci_low', 'ci_high')}
    release = load_json(root / 'docs/member1/dayC/releases/M2-REL-20261113-01/m2_release.json')
    st.info('Saved evidence · accepted LAV-DF research. No new inference or evaluation is executed.')
    try:
        if split == 'test':
            documented = documented_test_results((root / 'docs/member2/results/TEST_RESULTS.md').read_text())
            metadata = documented['metadata']
            st.caption(f"Recorded run: {metadata.get('run_id', 'see original report')} · split: test · sample count: {metadata.get('total_test_videos', 'unavailable')} · evaluation available")
        else:
            spec = load_json(root / 'configs/dayC/member1/experiment_spec.json')
            st.caption(f"LAV-DF · split: val · accepted cohort: {spec.get('validation_expected_total', 'unavailable')} clips · evaluation available")
        if split == 'test':
            for row in saved_result_table(root, 'test_manipulation_metrics.csv'):
                if row['mode'] in configs:
                    configs[row['mode']]['per_type'].setdefault(row['manipulation_type'], {})[row['metric']] = {
                        name: float(row[name]) for name in ('value', 'ci_low', 'ci_high')}
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f'Additional saved provenance unavailable: {exc}')
    st.caption('Configurations: ' + ', '.join(configs) + ' · recorded CPU benchmarks available separately')
    _performance(configs)
    _manipulations(_strata(configs))
    _decisions([])
    st.subheader('Computational cost')
    try:
        records = saved_result_table(root, 'cpu_model_runtime.csv')
        workload = st.selectbox('Recorded workload', list(dict.fromkeys(row['workload'] for row in records)), index=1, key='saved_workload')
        cost = [{'Configuration': row['mode'], 'Scope': 'cached_feature_model', 'Device': 'cpu',
                 'p50 (ms)': float(row['p50_ms']), 'p95 (ms)': float(row['p95_ms']), 'Peak memory': 'Not measured'}
                for row in records if row['workload'] == workload and row['mode'] in modes]
        st.dataframe(pd.DataFrame(cost), hide_index=True, use_container_width=True)
    except (OSError, ValueError) as exc:
        st.info(f'Computational cost unavailable: {exc}')
    st.caption('Recorded resident cached-feature model calls; excludes raw-media preprocessing. Simulated INT8 does not use native INT8 acceleration. CPU peak memory was not measured.')
    st.subheader('What the results mean')
    st.write('The accepted study does not establish statistically significant overall degradation from simulated low precision or a V/output restoration advantage. Intervals crossing zero do not prove equality. These findings apply to the recorded detector, intervention and cohort.')
    with st.expander('Technical details · saved provenance and original reports'):
        st.write(f"Original validation-fixed threshold: {release['fixed_threshold']}. Source-cluster bootstrap: {release['n_boot']} resamples, seed {release['seed']}.")
        st.caption('Recorded test aggregates are read only; no held-out rows, features or evaluator are opened by this view.')
        for mode in configs: st.caption(MODE_HELP[mode])
        for name in (overall_name, 'test_manipulation_metrics.csv', 'test_paired_comparisons.csv', 'cpu_model_runtime.csv', 'cpu_model_timing_samples.csv'):
            path = root / 'docs/results' / name
            if path.is_file():
                st.download_button('Download ' + name, path.read_bytes(), file_name=name, mime='text/csv', key='download_'+name)
        st.dataframe(pd.DataFrame(overall), hide_index=True, use_container_width=True)
        st.markdown('[Original research findings and limitations](https://github.com/SudarshanG9/AFML-project/blob/main/docs/06_EXPERIMENTS_AND_RESULTS.md)')


def _technical_run(root, result):
    from .run_inspection import verify_with_receipt, receipt_is_current
    run = Path(result['run_dir'])
    with st.expander('Technical details · provenance, integrity and exports'):
        st.caption('Finalization metadata is not an integrity check. Hash consistency does not authenticate authorship.')
        if st.button('Verify selected run integrity', key='workbench_verify'):
            try:
                st.session_state.setdefault('verified_runs', {})[str(run)] = verify_with_receipt(run)
            except Exception as exc:
                st.error(f'Integrity verification failed: {exc}')
        receipt = st.session_state.get('verified_runs', {}).get(str(run))
        if receipt:
            st.caption(f"Past check: {receipt['checked_at_utc']} · not continuous verification")
            if not receipt_is_current(run, receipt): st.warning('Files changed; verify again.')
            elif receipt['verdict'].get('ok'): st.success('Artifact integrity passed at the recorded check time.')
            else: st.error('Artifact integrity failed.')
            st.json(receipt['verdict'])
        else:
            st.caption('Not verified. Use the explicit integrity action when needed.')
        for name in ('execution_summary.json', 'failure_summary.json', 'run_manifest.json', 'provenance.json', 'resolved_config.json', 'environment_report.json', 'stage_profile.json'):
            if result.get(name) is not None:
                st.write('**' + name + '**')
                st.json(result[name])
        if result.get('metrics'):
            st.write('**Associated Member 2 evaluation and receipt**')
            st.json(result['metrics'])
            st.json(result['evaluation_receipt'])
        for path in result['available_artifacts']:
            try:
                if path.stat().st_size <= 32*1024*1024:
                    st.download_button('Download ' + path.name, path.read_bytes(), file_name=path.name, key='run_download_'+str(path))
            except OSError as exc:
                st.warning(f'Export unavailable for {path.name}: {exc}')



def _run(root, files):
    runs = [str(p) for p in files.get('run_dirs', [])]
    if not runs:
        st.info('No validation run artifacts discovered. Configure and execute an authorized validation experiment first.')
        return
    selected = st.session_state.get('selected_run')
    if selected not in runs: st.session_state['selected_run'] = runs[-1]
    run = st.selectbox('Validation run', runs, format_func=lambda value: Path(value).name, key='selected_run')
    try:
        result = load_experiment_results(root, run)
    except (OSError, ValueError) as exc:
        st.error(f'Run unavailable: {exc}')
        return
    summary = result.get('execution_summary.json') or {}
    st.info('New validation run · local inference artifacts; completion and evaluation are separate states.')
    st.caption(f"Run: {result['run_id']} · split: {result.get('split') or 'unavailable'} · origin: {summary.get('data_origin', 'unavailable')}")
    if result['completion_recorded']:
        st.write('**Inference completion recorded. Integrity verification is separate.**')
    else:
        st.warning('Run completion is not established. Available partial or failed artifacts remain inspectable.')
    st.caption('Configurations present: ' + (', '.join(result['available_modes']) or 'none'))
    st.caption('Sample count in validated predictions: ' + str(len({r['clip_id'] for r in result['predictions']})))
    for error in result['errors']: st.error(error)
    report = result['metrics']
    if report is None:
        st.info('Inference completed. Label-grounded evaluation has not been performed.' if result['completion_recorded'] else
                'Label-grounded evaluation has not been performed.')
    _performance(report.get('configs', {}) if report else {})
    _manipulations(_strata(report.get('configs', {})) if report else [])
    _decisions(result['predictions'], report.get('threshold') if report else None)
    st.subheader('Computational cost')
    if result['benchmarks']:
        cost = [{'Configuration': row['config'], 'Scope': row['scope'], 'Device': row['device'],
                 'Backend': row['backend'], 'p50 (ms)': row['p50_ms'], 'p95 (ms)': row['p95_ms'],
                 'Peak memory (bytes)': row['peak_memory_bytes'], 'Memory method': row['memory_method'],
                 'Input identity': row['workload'].get('input_identity_sha256')} for row in result['benchmarks']]
        st.dataframe(pd.DataFrame(cost), hide_index=True, use_container_width=True)
    else:
        st.info('No verified, identity-matched benchmark records available.')
    st.caption('Measurement scopes remain separate. Missing peak memory is unmeasured. Simulated INT8 is floating-point emulation; no integer-kernel speedup follows.')
    st.subheader('What the results mean')
    st.write('Descriptive observation of this run: score changes compare precision configurations, not ground-truth correctness. No automatic significance or restoration-superiority claim is made.')
    with st.expander('Label-grounded validation evaluation'):
        st.caption('Use an existing frozen protocol matching this exact validation-only manifest and checkpoint. The original validation-fixed threshold is preserved; no recalibration or test access is offered. Historical canonical-manifest protocols may not match a validation-only subset and will be refused.')
        protocol = st.text_input('Frozen protocol path', key='evaluation_protocol')
        if st.button('Evaluate validation labels', disabled=not (result['completion_recorded'] and bool(protocol.strip()))):
            with st.spinner('Running the existing Member 2 evaluator…'):
                outcome = evaluate_validation_results(root, run, protocol)
            st.session_state['evaluation_outcome'] = dict(code=outcome.code, message=outcome.diagnostics)
            if not outcome.code: st.rerun()
        outcome = st.session_state.get('evaluation_outcome')
        if outcome:
            (st.success if outcome['code'] == 0 else st.warning)(outcome['message'])
    _technical_run(root, result)


def render(root, files):
    st.subheader('Results')
    st.button('Refresh discovered artifacts', key='refresh_workbench')
    source = st.radio('Results source', ['Saved evidence', 'New validation run', 'Imported artifacts'], key='result_source', horizontal=True)
    if source == 'Saved evidence':
        _saved(root)
    elif source == 'New validation run':
        _run(root, files)
    else:
        from .ui import render_evaluation, render_benchmarks
        st.caption('Imported artifacts are shown independently. A selected report is not associated with any new run.')
        evaluation, benchmark = st.tabs(['Evaluation report', 'Benchmark records'])
        with evaluation:
            # Imported mode is explicit and cannot silently replace a report with saved tables.
            st.session_state['import_evaluation_source'] = 'Imported evaluation artifact'
            render_evaluation(root, files, imported_only=True)
        with benchmark:
            render_benchmarks(root, files, imported_only=True)
