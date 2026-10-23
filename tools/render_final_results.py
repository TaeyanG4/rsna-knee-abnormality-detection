"""Validate manual results and render English/Korean pages. No network or polling."""
import argparse
from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'docs/reports/final-results-data.json'
BASELINE = {
    'X': ('57415972', 'knee-lb-r6w958k1x3sr20-v2', '0.958'),
    'p25': ('57427331', 'knee-lb-r6w958k1n3s2p25-v1', '0.958'),
    'F': ('57475409', 'knee-lb-r6w958k1n3s2p25flatpv10-v1', '0.957'),
    'A5': ('57484279', 'knee-lb-r6w958k1n3s2p25a5v10-v1', '0.956'),
    'DF': ('57500332', 'knee-lb-r6w958k1n3s2p25a5flatpv10b-v1', '0.955'),
    'U': ('57510427', 'knee-lb-r6w958k1uxpv10-v1', None),
    'P4': ('57512878', 'knee-lb-r6w958k1n4fbp25v10-v1', None),
    'U4': ('57514281', 'knee-lb-r6w958k1u4fbv10-v1', None),
}
SNAPSHOT = 'https://github.com/TaeyanG4/rsna-knee-abnormality-detection/blob/4172ccf879887e3fa95f00cb714214e17a2805dd/'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def timestamp(value):
    require(isinstance(value, str), 'A KST timestamp is required')
    parsed = datetime.fromisoformat(value)
    require(parsed.utcoffset() == timedelta(hours=9), 'Timestamp must include +09:00')
    return parsed

def score(value):
    if value is None:
        return
    require(isinstance(value, str) and bool(re.fullmatch(r'(?:0(?:\.\d+)?|1(?:\.0+)?)', value)),
            'Scores must be exact display strings in [0,1]; unknown is null')
    number = Decimal(value)
    require(number.is_finite() and 0 <= number <= 1, 'Score outside [0,1]')

def evidence(row):
    timestamp(row['observed_at_kst'])
    require(isinstance(row['evidence'], str) and bool(row['evidence'].strip()), 'Evidence reference required')

def validate(d):
    require(d['schema_version'] == 1, 'Unsupported schema version')
    timestamp(d['prepared_at_kst'])
    cutoff = timestamp(d['baseline_cutoff_kst'])
    require(d['baseline_cutoff_kst'] == '2026-10-22T18:15:00+09:00', 'Frozen cutoff changed')
    candidates = d['candidates']
    ids = [x['id'] for x in candidates]
    require(set(BASELINE) <= set(ids) and 'F4' in ids and len(ids) == len(set(ids)), 'Missing/duplicate IDs')
    refs, observations = [], []
    changed = False
    for c in candidates:
        ref, version = c['submission_ref'], c['remote_version']
        if c['id'] in BASELINE:
            oldref, slug, oldscore = BASELINE[c['id']]
            require((ref, c['kernel'], version, c['baseline_public']) ==
                    (oldref, 'taeyangg4/' + slug, 1, oldscore), 'Frozen baseline identity/display changed')
        else:
            require(c['baseline_public'] is None, 'New candidates cannot acquire historical baseline scores')
        if ref is not None:
            require(isinstance(ref, str) and ref.isdigit(), 'Ref must be a digit string')
            require(type(version) is int and version > 0, 'Remote version must be positive integer')
            require(isinstance(c['kernel'], str) and bool(re.fullmatch(r'taeyangg4/[a-z0-9-]+', c['kernel'])), 'Invalid kernel')
            require(c['acceptance'] == 'accepted' and bool(c['acceptance_evidence']), 'Accepted identity needs receipt/evidence')
            timestamp(c['submitted_at_kst'])
            refs.append(ref)
            if c['id'] not in BASELINE:
                observations.append(timestamp(c['submitted_at_kst']))
                changed = True
        else:
            require(c['kernel'] is None and version is None and c['submitted_at_kst'] is None, 'Unaccepted identity must be null')
            require(c['acceptance'] == 'not_recorded', 'Unaccepted candidate cannot be marked accepted')
        score(c['baseline_public'])
        r = c['result']
        require(r['inference_status'] in ['not_recorded', 'running', 'complete', 'error', 'cancelled'], 'Invalid inference status')
        score(r['public_score'])
        score(r['private_score'])
        observed = any(r[k] is not None for k in ['public_score', 'private_score', 'observed_at_kst', 'evidence']) or r['inference_status'] != 'not_recorded'
        if observed:
            require(ref is not None, 'Do not assign results to an unaccepted candidate')
            evidence(r)
            observations.append(timestamp(r['observed_at_kst']))
            changed = True
            if r['public_score'] is not None or r['private_score'] is not None:
                require(r['inference_status'] == 'complete', 'Scores require confirmed complete inference')
    require(len(refs) == len(set(refs)), 'Duplicate submission references')
    s = d['final_selection']
    require(s['state'] in ['unconfirmed', 'confirmed'], 'Invalid selection state')
    if s['state'] == 'confirmed':
        require(len(s['candidate_ids']) == 2 and len(set(s['candidate_ids'])) == 2, 'Final pair needs two distinct candidates')
        lookup = {c['id']: c for c in candidates}
        for key in s['candidate_ids']:
            require(key in lookup and lookup[key]['submission_ref'] is not None, 'Selected candidate needs accepted identity')
        evidence(s)
        observations.append(timestamp(s['observed_at_kst']))
        changed = True
    else:
        require(s['candidate_ids'] == [] and s['observed_at_kst'] is None and s['evidence'] is None, 'Unconfirmed pair must be empty')
    o = d['competition_outcome']
    require(o['state'] in ['pending', 'provisional', 'final'], 'Invalid competition outcome state')
    score(o['private_score'])
    for field in ['rank', 'total_teams']:
        require(o[field] is None or (type(o[field]) is int and o[field] > 0), 'Rank/team count must be positive integers or null')
    if o['rank'] is not None and o['total_teams'] is not None:
        require(o['rank'] <= o['total_teams'], 'Rank exceeds team count')
    if o['state'] == 'pending':
        require(all(o[k] is None for k in ['private_score', 'rank', 'total_teams', 'award', 'observed_at_kst', 'evidence']), 'Pending outcome contains results')
    else:
        evidence(o)
        observations.append(timestamp(o['observed_at_kst']))
        changed = True
        require(o['private_score'] is not None or o['rank'] is not None, 'Outcome needs score or rank')
        if o['state'] == 'final':
            require(o['private_score'] is not None and o['rank'] is not None, 'Final outcome needs score and rank')
    for item in d['change_log']:
        timestamp(item['at_kst'])
        require(bool(item['source']) and bool(item['change_en']) and bool(item['change_ko']), 'Change log needs source and bilingual description')
    for item in d['interpretation']:
        require(bool(item['text_en']) and bool(item['text_ko']) and bool(item['evidence']), 'Interpretation needs bilingual text and evidence')
    changed = changed or bool(d['interpretation'])
    if changed:
        require(d['updated_at_kst'] is not None and bool(d['change_log']), 'New results need update time and change log')
        updated = timestamp(d['updated_at_kst'])
        require(updated >= cutoff and all(cutoff <= x <= updated for x in observations), 'Invalid observation chronology')
    elif d['updated_at_kst'] is not None:
        timestamp(d['updated_at_kst'])

def cell(value, missing='—'):
    if value is None:
        return missing
    return str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', ' ')

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in rows])

def render(d, ko=False):
    t = lambda en, kr: kr if ko else en
    o, s = d['competition_outcome'], d['final_selection']
    state = {'pending': t('Awaiting results', '결과 대기'), 'provisional': t('Provisional result', '잠정 결과'), 'final': t('Final result recorded', '최종 결과 기록')}[o['state']]
    lines = [
        t('# Final results — RSNA Knee Abnormality Detection', '# 최종 결과 — RSNA 무릎 이상 탐지'), '',
        t('**English main** · [한국어](final-results.ko.md) · [Pre-result snapshot](project-summary-20261022.md)',
          '[English main](final-results.md) · **한국어** · [결과 발표 전 기록](project-summary-20261022.ko.md)'), '',
        '**' + state + '**', '',
        t('Prepared: ', '준비 시각: ') + d['prepared_at_kst'] + '. ' + t('Latest verified result update: ', '마지막 결과 확인: ') + cell(d['updated_at_kst'], t('none yet.', '아직 없음.')), '',
        t('The expected date, 2026-10-23, was supplied by Taeyang; release time is unconfirmed. This page accepts manually supplied results and does not poll scores.',
          '결과 예정일 2026-10-23은 Taeyang이 알려 준 날짜이며 발표 시각은 미확인입니다. 실제 결과를 받아 기록할 준비 페이지이며 점수를 자동 조회하지 않습니다.'), '',
        t('## 1. Competition outcome', '## 1. 대회 결과'), '',
        table([t('Field', '항목'), t('Recorded value', '기록')], [
            [t('Result state', '결과 상태'), state], [t('Private score', '비공개 점수'), o['private_score']],
            [t('Rank', '순위'), o['rank']], [t('Total teams', '전체 팀 수'), o['total_teams']], [t('Award / medal', '수상 / 메달'), o['award']],
            [t('Observed at (KST)', '확인 시각 (KST)'), o['observed_at_kst']], [t('Evidence / source', '근거 / 출처'), o['evidence']]]), '',
        t('A dash means unrecorded, never zero or failure. Overall competition score/rank is separate from individual submission scores; it is not calculated as an average of the final pair.',
          '대시는 미기록을 뜻하며 0점이나 실패를 뜻하지 않습니다. 대회 전체 점수·순위와 개별 제출 점수를 구분합니다. 최종 두 제출물의 평균을 대회 점수로 계산하지 않습니다.'), '',
        t('## 2. Confirmed final submissions', '## 2. 확정된 최종 제출물'), '',
        t('Selection state: **' + s['state'] + '**. The pre-result default X + F is a policy fallback, not confirmation of actual selection.',
          '선택 상태: **' + ('확정' if s['state'] == 'confirmed' else '미확정') + '**. 기본 조합 X + F는 정책상 기본값이며 실제 선택 완료를 뜻하지 않습니다.'), '']
    lookup = {c['id']: c for c in d['candidates']}
    chosen = [lookup[k] for k in s['candidate_ids']]
    lines += [table([t('Slot', '슬롯'), t('Candidate', '후보'), 'Submission ref', t('Remote version', '원격 버전')],
        [[str(i + 1), chosen[i]['id'] if i < len(chosen) else None, chosen[i]['submission_ref'] if i < len(chosen) else None, chosen[i]['remote_version'] if i < len(chosen) else None] for i in range(2)]), '',
        t('Selection confirmation time / evidence: ', '선택 확인 시각 / 근거: ') + cell(s['observed_at_kst']) + ' / ' + cell(s['evidence']), '',
        t('Use the [policy frozen before outcomes](' + SNAPSHOT + 'plan.md), Updates 7–10. Eligibility at 2026-10-23 08:00 KST and the recorded 08:59 close are separate from result publication. This page does not choose or change the pair.',
          '[결과 전 고정 정책](' + SNAPSHOT + 'plan.md) Updates 7–10을 따릅니다. 2026-10-23 08:00 KST의 후보 판단 시각과 기록된 08:59 마감은 최종 결과 발표와 별개입니다. 이 페이지는 조합을 자동 선택하거나 변경하지 않습니다.'), '',
        t('## 3. Submission results', '## 3. 제출별 결과'), '',
        t('Baseline public displays are frozen at **2026-10-22 18:15 KST**. New observations occupy separate columns. Acceptance is not proof of hidden completion.',
          '기준 공개 표시값은 **2026-10-22 18:15 KST** 기록입니다. 이후 결과는 별도 열에 기록합니다. 접수 사실만으로 비공개 추론 완료를 단정하지 않습니다.'), '']
    statuses = {'not_recorded': t('Not recorded', '미기록'), 'running': t('Running', '실행 중'), 'complete': t('Complete', '완료'), 'error': t('Error', '오류'), 'cancelled': t('Cancelled', '취소')}
    lines += [table([t('Candidate', '후보'), 'Ref / v', t('Public at cutoff', '기준 공개'), t('New inference status', '새 추론 상태'), t('New public', '새 공개'), t('Private', '비공개')],
        [[c['id'], c['submission_ref'] + ' / v' + str(c['remote_version']) if c['submission_ref'] else None, c['baseline_public'], statuses[c['result']['inference_status']], c['result']['public_score'], c['result']['private_score']] for c in d['candidates']]), '',
        t('F4 was preparation-only at cutoff. Fill its exact accepted kernel/ref/version from a receipt before adding results. Recovery submissions get separate rows; historical identities are not overwritten.',
          'F4는 기준 시점에 준비 단계였습니다. 접수 원장으로 정확한 커널/ref/버전을 확인한 뒤 결과를 추가합니다. 복구 제출물은 별도 행에 기록하고 기존 제출 식별자를 덮어쓰지 않습니다.'), '',
        t('### Exact identities and evidence', '### 정확한 식별자와 근거'), '',
        table([t('Candidate', '후보'), 'Kernel', t('Accepted at (KST)', '접수 시각 (KST)'), t('Baseline evidence', '기준 근거')],
              [[c['id'], c['kernel'], c['submitted_at_kst'], c['baseline_evidence']] for c in d['candidates']]), '',
        table([t('Candidate', '후보'), t('Observed at (KST)', '관측 시각 (KST)'), t('New result evidence', '새 결과 근거')],
              [[c['id'], c['result']['observed_at_kst'], c['result']['evidence']] for c in d['candidates']]), '',
        t('## 4. Interpretation after results', '## 4. 결과 확인 후 해석'), '']
    if d['interpretation']:
        lines += ['- ' + cell(x['text_ko' if ko else 'text_en']) + ' (' + cell(x['evidence']) + ')' for x in d['interpretation']]
    else:
        lines += [t('No post-result conclusion recorded. Examine these comparisons only when comparable, correctly identified results are available:',
                    '결과에 근거한 결론은 아직 없습니다. 식별자와 평가 조건을 확인한 뒤 다음 비교를 기록합니다:'), '',
            table([t('Question', '질문'), t('Comparison', '비교 대상'), t('Conclusion', '결론')], [
                [t('Does full-bag addition transfer?', 'Full-bag 추가 효과가 이어졌는가'), 'P4 vs p25; U4 vs U', t('Awaiting evidence', '근거 대기')],
                [t('Does flat P help?', 'Flat P 보완 조합이 도움이 되었는가'), 'F vs p25; F4 vs P4', t('Awaiting evidence', '근거 대기')],
                [t('Does union fusion improve?', 'Union 결합이 개선되었는가'), 'U vs X / p25', t('Awaiting evidence', '근거 대기')]])]
    lines += ['', t('These are descriptive comparisons, not controlled causal estimates: runtime/stride differences and rounded displays may matter. Keep public, private and exposed local evidence distinct; preserve negative findings and review states.',
                    '관측값의 비교이며 통제된 인과 효과 추정이 아닙니다. 실행 방식·stride·반올림 표시 차이가 영향을 줄 수 있습니다. 공개·비공개·노출된 로컬 근거를 구분하고 기존 실패와 검토 상태를 보존합니다.'), '',
        t('## 5. Update log', '## 5. 갱신 이력'), '']
    lines += [table([t('Time (KST)', '시각 (KST)'), t('Source', '출처'), t('Change', '변경 내용')],
              [[x['at_kst'], x['source'], x['change_ko' if ko else 'change_en']] for x in d['change_log']])
              if d['change_log'] else t('No result updates yet. Page preparation is not a result event.', '아직 결과 갱신이 없습니다. 페이지 준비는 결과 관측이 아닙니다.')]
    fence = chr(96) * 3
    lines += ['', t('## 6. Update this page', '## 6. 이 페이지 갱신 방법'), '',
        t('Edit [final-results-data.json](final-results-data.json) once, then run:',
          '[final-results-data.json](final-results-data.json)을 한 번 수정한 뒤 실행합니다:'), '',
        fence + 'powershell\npython tools/render_final_results.py\npython tools/render_final_results.py --check\n' + fence, '',
        t('Both languages regenerate from one record. See the [update guide](final-results-update-guide.md) for fields, evidence and guarded publication.',
          '하나의 기록에서 영어·한국어 두 페이지를 함께 갱신합니다. 입력 항목·근거·게시 절차는 [갱신 안내](final-results-update-guide.md)에 있습니다.'), '',
        t('[Immutable pre-result report](' + SNAPSHOT + 'docs/reports/project-summary-20261022.md) · [Documentation audit](documentation-audit-20261022.md)',
          '[시점이 고정된 연구 보고서](' + SNAPSHOT + 'docs/reports/project-summary-20261022.ko.md) · [문서 검수 기록](documentation-audit-20261022.md)'), '',
        '<!-- Generated by tools/render_final_results.py; edit final-results-data.json instead. -->', '']
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8-sig'))
    validate(data)
    out = args.output_dir or args.data.parent
    for ko, name in [(False, 'final-results.md'), (True, 'final-results.ko.md')]:
        content, path = render(data, ko), out / name
        if args.check:
            require(path.exists() and path.read_text(encoding='utf-8') == content, 'Generated page out of date: ' + name)
        else:
            out.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8', newline='\n')
    print(json.dumps({'status': 'PASS', 'mode': 'check' if args.check else 'render', 'candidates': len(data['candidates']), 'outcome': data['competition_outcome']['state'], 'selection': data['final_selection']['state'], 'network': False}))

if __name__ == '__main__':
    main()
