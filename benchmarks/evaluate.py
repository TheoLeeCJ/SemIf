"""Every declared row counts; missing, invalid and unparsed outputs remain failures."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random


def indexed(rows, name):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise ValueError(f'Duplicate ID in {name}: {row["id"]}')
        result[row['id']] = row
    return result


def read_jsonl(path):
    with open(path) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    indexed(rows, str(path))
    return rows


def vector(values, ids):
    if isinstance(values, dict):
        if set(values) != set(ids):
            raise ValueError('Probability keys differ from gold option IDs')
        values = [values[key] for key in ids]
    if not isinstance(values, list) or len(values) != len(ids):
        raise ValueError('Wrong probability vector length/type')
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in values):
        raise ValueError('Invalid/nonfinite probability')
    if abs(sum(values)-1) > 1e-4:
        raise ValueError('Probabilities do not sum to one')
    return values


def align(gold, predictions):
    truth, outputs = indexed(gold, 'gold'), indexed(predictions, 'predictions')
    if outputs.keys() - truth.keys():
        raise ValueError(f'Unknown prediction IDs: {sorted(outputs.keys()-truth.keys())[:5]}')
    rows = []
    for item in gold:
        ids = [o['id'] for o in item['options']]
        if len(set(ids)) != len(ids) or not 0 <= item['label'] < len(ids):
            raise ValueError(f'Invalid gold options/label: {item["id"]}')
        target = item.get('target_distribution')
        if target is not None:
            target = vector(target, ids)
        row = dict(id=item['id'], group_id=item['group_id'], family=item['family'],
                   gold_id=ids[item['label']], predicted_id=None, correct=False,
                   confidence=None, nll=None, brier=None, status='missing', probabilities=None)
        pred = outputs.get(item['id'])
        if pred is not None:
            try:
                if pred.get('parse_status') == 'unparsed' or pred.get('error'):
                    raise ValueError(pred.get('parse_error') or pred.get('error') or 'Unparsed output')
                values = pred.get('probabilities')
                if values is None:
                    choice = pred.get('prediction_id')
                    if choice not in ids:
                        raise ValueError('Missing or out-of-set native prediction')
                    row.update(status='native_decision', predicted_id=choice)
                else:
                    if isinstance(values, list) and 'option_ids' in pred:
                        pred_ids = pred['option_ids']
                        if len(pred_ids) != len(values) or len(set(pred_ids)) != len(pred_ids):
                            raise ValueError('Invalid prediction option IDs')
                        values = dict(zip(pred_ids, values))
                    ps = vector(values, ids)
                    chosen = max(range(len(ps)), key=lambda k: ps[k])
                    row.update(status='distribution', predicted_id=ids[chosen], probabilities=ps,
                               confidence=ps[chosen], nll=-math.log(max(ps[item['label']], 1e-12)),
                               brier=sum((p-(k == item['label']))**2 for k,p in enumerate(ps)))
                    if target is not None:
                        row['analytic'] = dict(squared_probability_error=sum((p-q)**2 for p,q in zip(ps,target)),
                            expected_nll=-sum(q*math.log(max(p,1e-12)) for p,q in zip(ps,target)),
                            expected_brier=1+sum(p*p-2*p*q for p,q in zip(ps,target)))
                row['correct'] = row['predicted_id'] == row['gold_id']
            except (ValueError, TypeError) as exc:
                row.update(status='invalid', error=str(exc))
        rows.append(row)
    return rows


def clusters(rows, key='group_id'):
    result = defaultdict(list)
    for row in rows:
        result[row[key]].append(row)
    return result


def basic(rows):
    if not rows:
        return dict(n=0, accuracy=None, balanced_accuracy=None, macro_f1=None)
    recalls, f1 = [], []
    for label in sorted({r['gold_id'] for r in rows}):
        tp = sum(r['gold_id']==label and r['predicted_id']==label for r in rows)
        fp = sum(r['gold_id']!=label and r['predicted_id']==label for r in rows)
        fn = sum(r['gold_id']==label and r['predicted_id']!=label for r in rows)
        recalls.append(tp/(tp+fn))
        f1.append(2*tp/(2*tp+fp+fn))
    return dict(n=len(rows), accuracy=sum(r['correct'] for r in rows)/len(rows),
                balanced_accuracy=sum(recalls)/len(recalls), macro_f1=sum(f1)/len(f1))


def summarize(rows):
    result = basic(rows)
    if not rows:
        return result
    groups = list(clusters(rows).values())
    valid = [r for r in rows if r['status']=='distribution']
    result.update(source_groups=len(groups), invalid_or_missing=sum(r['status'] in ('invalid','missing') for r in rows),
                  all_decisions_correct_group_rate=sum(all(r['correct'] for r in g) for g in groups)/len(groups),
                  probability_rows=len(valid), probability_coverage=len(valid)/len(rows))
    for key in ('nll','brier'):
        value = sum(r[key] for r in valid)/len(valid) if valid else None
        result[key] = value if len(valid)==len(rows) else None
        result[key+'_valid_distributions_only'] = value
    result['nll_probability_floor'] = 1e-12
    ranked = sorted(valid, key=lambda r: (-r['confidence'],r['id']))
    result['risk_coverage'] = []
    for fraction in (.25,.5,.75,1):
        if not ranked:
            break
        k = min(len(ranked), max(1,math.ceil(len(rows)*fraction)))
        threshold = ranked[k-1]['confidence']
        accepted = [r for r in ranked if r['confidence']>=threshold]
        result['risk_coverage'].append(dict(requested_coverage=fraction, coverage=len(accepted)/len(rows),
            n=len(accepted), threshold=threshold, error=1-sum(r['correct'] for r in accepted)/len(accepted)))
    result['reliability_bins'] = []
    for b in range(10):
        part = [r for r in valid if min(9,int(r['confidence']*10))==b]
        if part:
            result['reliability_bins'].append(dict(lower=b/10,upper=(b+1)/10,n=len(part),
                mean_confidence=sum(r['confidence'] for r in part)/len(part),
                accuracy=sum(r['correct'] for r in part)/len(part)))
    rng, values = random.Random(217), []
    for _ in range(1000):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        values.append(sum(r['correct'] for g in draw for r in g)/sum(map(len,draw)))
    values.sort()
    result['accuracy_cluster_bootstrap_95'] = [values[25],values[974]]
    return result


def balanced_metric(rows):
    families = clusters(rows,'family')
    return sum(basic(part)['balanced_accuracy'] for part in families.values())/len(families)


def paired_comparison(left, right, samples=1000, seed=217):
    """Use identical source-group draws on both systems, stratified by family."""
    a,b = indexed(left,'left'),indexed(right,'right')
    if a.keys()!=b.keys() or not a or samples < 40:
        raise ValueError('Need same nonempty gold IDs and at least 40 bootstrap samples')
    for key in a:
        if any(a[key][f]!=b[key][f] for f in ('group_id','family','gold_id')):
            raise ValueError('Paired gold metadata disagree')
    strata = defaultdict(dict)
    for group,rows in clusters(left).items():
        families = {r['family'] for r in rows}
        if len(families)!=1:
            raise ValueError('Source group crosses families')
        strata[next(iter(families))][group] = [r['id'] for r in rows]
    rng,draws = random.Random(seed),[]
    for _ in range(samples):
        ids = []
        for groups in strata.values():
            units = list(groups.values())
            for _ in units:
                ids.extend(units[rng.randrange(len(units))])
        draws.append(balanced_metric([a[k] for k in ids])-balanced_metric([b[k] for k in ids]))
    draws.sort()
    return dict(metric='mean_family_balanced_accuracy', difference=balanced_metric(left)-balanced_metric(right),
        paired_source_group_bootstrap_95=[draws[int(.025*samples)],draws[min(samples-1,int(.975*samples))]],
        source_groups=sum(map(len,strata.values())),samples=samples,seed=seed,
        method='Paired source-group bootstrap stratified by family; represented gold classes per draw')


def evaluate(gold,predictions,comparison=None):
    rows = align(gold,predictions)
    families = {name:summarize(part) for name,part in clusters(rows,'family').items()}
    analytic = [dict(id=r['id'],**r['analytic']) for r in rows if 'analytic' in r]
    result = dict(available_gold=len(gold),scored=len(predictions),evaluated=len(rows),
        coverage=sum(r['status'] not in ('invalid','missing') for r in rows)/len(rows) if rows else 0,
        missing=sum(r['status']=='missing' for r in rows),invalid=sum(r['status']=='invalid' for r in rows),
        family_results=families,mean_family_balanced_accuracy=balanced_metric(rows) if rows else None,
        mean_family_macro_f1=sum(r['macro_f1'] for r in families.values())/len(families) if families else None,
        analytic_uncertainty={'n':len(analytic),'rows':analytic},errors=[r for r in rows if not r['correct']],
        limitations=['All gold rows count in accuracy, including missing/invalid/unparsed predictions.',
                    'Native decisions have no probability estimates; one-hot distributions are not fabricated.',
                    'Risk/coverage is descriptive; tied confidence values remain together.',
                    'Analytic event hard-label accuracy measures modal-class agreement, not observed-event accuracy.'])
    if comparison is not None:
        result['paired_comparison'] = paired_comparison(rows,align(gold,comparison))
    return result


def calibrated_threshold_gate(gold, predictions, threshold=0.8, abstain_option=None):
    """Generic auto-decide-versus-review gate on top-label confidence.

    Intended for temperature-scaled prediction files (see docs/CALIBRATION.md). Unlike
    ``screening_gate``, this does not require the frozen 96-row authored falsification
    screen and does not change that screen's semantics.
    """
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 < threshold <= 1:
        raise ValueError('calibrated threshold must be in (0, 1]')
    rows = align(gold, predictions)
    preds = indexed(predictions, 'predictions')
    temperatures = []
    for row in rows:
        pred = preds.get(row['id'])
        cal = (pred or {}).get('calibration') or {}
        if cal.get('method') == 'temperature_scaling' and 'temperature' in cal:
            temperatures.append(cal['temperature'])
    decisions = []
    automatic = review = automatic_correct = 0
    for row in rows:
        available = row['status'] == 'distribution' and row['confidence'] is not None
        accept = available and row['confidence'] >= threshold
        if abstain_option is not None and row['predicted_id'] == abstain_option:
            accept = False
        disposition = 'automatic_semantic_decision' if accept else 'review'
        if accept:
            automatic += 1
            automatic_correct += bool(row['correct'])
        else:
            review += 1
        decisions.append(dict(id=row['id'], choice=row['predicted_id'], confidence=row['confidence'],
                              disposition=disposition, correct=row['correct'], status=row['status']))
    unique_t = sorted({t for t in temperatures})
    return dict(
        policy=dict(version='calibrated-threshold-v1', threshold=threshold,
                    abstain_option=abstain_option,
                    scope='any hard-label workload; not the frozen 96-row screen'),
        rows=len(rows),
        automatic_decisions=automatic,
        review_decisions=review,
        automatic_accuracy=(automatic_correct / automatic) if automatic else None,
        coverage=automatic / len(rows) if rows else 0.0,
        temperature_scaling_rows=len(temperatures),
        applied_temperatures=unique_t,
        calibrated_predictions=len(temperatures) == len(predictions) and len(unique_t) <= 1,
        decisions=decisions,
        limitation='A threshold on temperature-scaled confidence is an operating point, '
                   'not Jev-level operational calibration.',
    )


def screening_gate(gold, predictions, policy='distribution', threshold=0.8):
    """Frozen semantic-decision gate; not a benchmark of executed workflow actions."""
    if policy not in ('distribution', 'native') or threshold != 0.8:
        raise ValueError('Screen policy is frozen: distribution p>=0.8 or native parsed decisions')
    rows = align(gold, predictions)
    result_by_id = indexed(rows, 'aligned')
    groups = clusters(gold)
    family_counts, family_pairs = defaultdict(int), defaultdict(int)
    issues, pair_correct, missing_automatic, original_correct_automatic = [], 0, 0, 0
    missing_count, evidence_correct, original_automatic = 0, 0, 0
    decisions = []
    required = {'original', 'criterion_reversal', 'evidence_change', 'missing'}
    for group, items in groups.items():
        variants = {}
        for item in items:
            variant = item.get('provenance', {}).get('variant')
            if variant in variants:
                issues.append(f'{group}: duplicate variant {variant}')
            variants[variant] = item
        if set(variants) != required or len(items) != 4 or len({r['family'] for r in items}) != 1:
            issues.append(f'{group}: requires four declared variants in one family')
            continue
        family = items[0]['family']
        family_counts[family] += 1
        paired = all(result_by_id[variants[v]['id']]['correct'] for v in ('original','criterion_reversal'))
        pair_correct += paired
        family_pairs[family] += paired
        evidence_correct += result_by_id[variants['evidence_change']['id']]['correct']
        for variant, item in variants.items():
            row = result_by_id[item['id']]
            semantic_choice = row['predicted_id']
            available = row['status'] == ('distribution' if policy == 'distribution' else 'native_decision')
            automatic = available and semantic_choice != 'insufficient'
            if policy == 'distribution':
                automatic = automatic and row['confidence'] >= threshold
            decisions.append(dict(id=row['id'], variant=variant, choice=semantic_choice,
                                  disposition='automatic_semantic_decision' if automatic else 'review',
                                  correct=row['correct']))
            if variant == 'missing':
                missing_count += 1
                if row['gold_id'] != 'insufficient':
                    issues.append(f'{item["id"]}: missing variant must have insufficient gold')
                missing_automatic += automatic
            elif variant == 'original':
                original_automatic += automatic
                original_correct_automatic += automatic and row['correct']
    expected_families = {'evidence_interpretation','rule_application','candidate_selection'}
    if len(gold)!=96 or len(groups)!=24 or set(family_counts)!=expected_families or any(n!=8 for n in family_counts.values()):
        issues.append('Screen requires 96 rows, 24 groups and eight groups in each declared family')
    complete = all(r['status'] == ('distribution' if policy == 'distribution' else 'native_decision') for r in rows)
    checks = dict(schema=not issues, complete_valid_outputs=complete,
                  original_reversal_pairs=pair_correct>=20,
                  every_family_pairs=set(family_counts)==expected_families and all(family_pairs[f]>=6 for f in expected_families),
                  no_unsupported_missing_decision=missing_count==24 and missing_automatic==0,
                  useful_original_coverage=original_correct_automatic>=12)
    return dict(targets_met=all(checks.values()), structurally_valid=not issues,
                decision='Reviewer decides whether observed failures justify a bounded pilot; numerical near-misses are not automatic vetoes.',
                checks=checks, schema_errors=issues,
                policy=dict(version='screen-policy-v1', mode=policy,
                            threshold=threshold if policy=='distribution' else None,
                            abstain_option='insufficient', scope='semantic decisions, not executed workflow actions'),
                original_reversal_pairs_correct=pair_correct, family_pairs_correct=dict(family_pairs),
                evidence_change_correct=evidence_correct, missing_cases=missing_count,
                unsupported_missing_decisions=missing_automatic,
                original_automatic_decisions=original_automatic,
                original_correct_automatic_decisions=original_correct_automatic,
                accuracy_including_abstentions=basic(rows)['accuracy'], decisions=decisions,
                limitation='Small falsification screen; zero errors does not certify safety/calibration.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gold','predictions','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--comparison')
    parser.add_argument('--screen-policy', choices=('distribution','native'))
    parser.add_argument('--calibrated-threshold', type=float,
                        help='Generic auto-decide vs review gate on top-label confidence (any workload).')
    parser.add_argument('--abstain-option',
                        help='Option id that is never auto-decided under --calibrated-threshold.')
    args=parser.parse_args()
    if args.abstain_option is not None and args.calibrated_threshold is None:
        parser.error('--abstain-option requires --calibrated-threshold')
    gold, predictions = read_jsonl(args.gold), read_jsonl(args.predictions)
    result=evaluate(gold, predictions, read_jsonl(args.comparison) if args.comparison else None)
    if args.screen_policy:
        result['screening_gate'] = screening_gate(gold, predictions, args.screen_policy)
    if args.calibrated_threshold is not None:
        result['calibrated_threshold_gate'] = calibrated_threshold_gate(
            gold, predictions, args.calibrated_threshold, args.abstain_option)
    with Path(args.output).open('x') as destination:
        destination.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('scored','coverage','mean_family_balanced_accuracy')}))
