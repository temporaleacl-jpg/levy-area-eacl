"""
src/preprocessing.py

Corpus-specific session filtering and feature extraction for the
Lévy-area orientation-coupling study.

Two independent pipelines:
  1. Korean counseling corpus (AI Hub #71806)
  2. Supreme Court oral arguments (ConvoKit), advocate-only filtering

Each produces a `bal_features`-style dict keyed by session ID, or
(for Supreme Court) a list of pre-filtered session dicts, ready to
be consumed by src/orientation_tests.py and
src/exploratory_clustering.py.

IMPORTANT (see paper §3.2, footnote): an earlier version of the
Korean pipeline used a pre-computed, fixed-length `levy_area` array
per session and indexed into it with `peak // 10`. This silently
dropped any session whose risk turn fell beyond the array's length
(an off-by-one boundary condition), which excluded one valid
session (dialogue ID A105) from the reported N=70. The corrected
Korean corpus size is N=71. `compute_levy_features_window_v2`
below recomputes Lévy-area windows directly from each session's
full embedding sequence rather than indexing into a fixed-length
cache, and is therefore not subject to this bug.
"""

import os
import re
import json
import zipfile
import pickle

import numpy as np
from sklearn.decomposition import PCA

from levy_area import compute_levy_area_2d

# ============================================================
# SECTION 1 — Korean counseling corpus (AI Hub #71806)
# ============================================================

RISK_SYMPTOMS = ['suicidal', 'trauma_experience', 'loss_of_control']

# Turns that are questionnaire/scale responses rather than
# spontaneous risk disclosures (see paper §3.2: "89
# questionnaire-response turns removed").
SCALE_CLIENT_PATTERNS = [
    r'^[0-9]\s*점[이요]*\s*[.]?$',
    r'^보통이다\s*[.]?$',
    r'^조금 아니다\s*[.]?$',
    r'^아니다\s*[.]?$',
    r'^그렇다\s*[.]?$',
    r'^[0-9]\s*$',
]
SCALE_COUNSELOR_PATTERNS = [
    r'특정 행위',
    r'[0-9]+번[.,]\s*(이전보다|나는|특정|뭔가)',
    r'나는.*할 수 있다',
    r'PHQ|BDI|척도|설문',
]


def is_scale_client(text):
    text = str(text).strip()
    return any(re.match(p, text) for p in SCALE_CLIENT_PATTERNS)


def is_scale_counselor(text):
    text = str(text).strip()
    return any(re.search(p, text) for p in SCALE_COUNSELOR_PATTERNS)


def load_korean_sessions_raw(base_dir):
    """
    Loads raw AI Hub session JSONs from zip archives in `base_dir`.

    Each session JSON has the structure:
        {
          "id": <session_id>,
          "class": <diagnostic category>,
          "paragraph": [
            {"index": int, "paragraph_speaker": "내담자"|"상담사",
             "paragraph_text": str,
             "suicidal": 0|1, "trauma_experience": 0|1,
             "loss_of_control": 0|1},
            ...
          ]
        }

    Returns a dict: {session_id: {"category": str, "turns": [...]}}
    where each turn is {"speaker": "counselor"|"client",
    "text": str, "risk": 0|1}.
    """
    sessions = {}
    all_zips = sorted(f for f in os.listdir(base_dir) if f.endswith('.zip'))
    print(f"전체 zip 파일: {len(all_zips)}개")

    for zip_name in all_zips:
        zip_path = os.path.join(base_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'r') as z:
            for json_name in z.namelist():
                if not json_name.endswith('.json'):
                    continue
                try:
                    with z.open(json_name) as f:
                        raw = f.read()
                    try:
                        data = json.loads(raw.decode('utf-8'))
                    except UnicodeDecodeError:
                        data = json.loads(raw.decode('cp949'))
                except Exception as e:
                    print(f"  ⚠️ 읽기 실패: {json_name} ({e})")
                    continue

                sid = data.get('id', '')
                category = data.get('class', '')
                paragraphs = sorted(
                    data.get('paragraph', []),
                    key=lambda x: x.get('index', 0)
                )

                turns = []
                for turn in paragraphs:
                    speaker_raw = turn.get('paragraph_speaker')
                    speaker = 'client' if speaker_raw == '내담자' else 'counselor'
                    text = turn.get('paragraph_text', '')

                    is_risk = any(turn.get(r, 0) == 1 for r in RISK_SYMPTOMS)

                    # Questionnaire-response filtering (turn-level):
                    # a nominally "risk" turn that is actually a scale
                    # response is downgraded to non-risk, following
                    # the same logic applied to both speaker roles.
                    if is_risk:
                        if speaker == 'client' and is_scale_client(text):
                            is_risk = False
                        elif speaker == 'counselor' and is_scale_counselor(text):
                            is_risk = False

                    turns.append({
                        'speaker': speaker,
                        'text': text,
                        'risk': 1 if is_risk else 0,
                    })

                sessions[sid] = {'category': category, 'turns': turns}

    print(f"세션 수: {len(sessions)}")
    return sessions


def extract_client_turns_balanced(session):
    """
    Extracts client turns with the preceding counselor turn prepended
    as context (truncated to 200 characters), separated by [SEP].
    """
    turns = session['turns']
    result = []
    for i, turn in enumerate(turns):
        if turn['speaker'] != 'client':
            continue
        prev_counselor = ""
        for j in range(i - 1, -1, -1):
            if turns[j]['speaker'] == 'counselor':
                prev_counselor = turns[j]['text'][:200]
                break
        input_text = (prev_counselor + " [SEP] " + turn['text']
                      if prev_counselor else turn['text'])
        result.append({
            'turn_idx': i,
            'input_text': input_text,
            'risk': turn['risk'],
        })
    return result


def build_korean_bal_features(base_dir, min_client_turns=10,
                               model_name='sentence-transformers/LaBSE',
                               batch_size=32):
    """
    Full Korean-corpus pipeline: raw JSON -> questionnaire filtering
    -> client-turn extraction -> LaBSE embedding -> bal_features dict.

    Only sessions with at least one non-questionnaire risk turn are
    retained downstream (enforced later, in
    compute_levy_features_window_v2, not here) -- this function
    keeps all sessions with >= min_client_turns client turns and
    lets the risk-turn requirement be applied at window-construction
    time, exactly as in the original pipeline.
    """
    from sentence_transformers import SentenceTransformer

    raw_sessions = load_korean_sessions_raw(base_dir)

    print("LaBSE 로딩...")
    model = SentenceTransformer(model_name)

    bal_features = {}
    n_questionnaire_removed = 0
    for i, (sid, session) in enumerate(raw_sessions.items()):
        client_turns = extract_client_turns_balanced(session)
        if len(client_turns) < min_client_turns:
            continue

        texts = [t['input_text'] for t in client_turns]
        embs = model.encode(texts, batch_size=batch_size,
                             show_progress_bar=False,
                             normalize_embeddings=True)

        bal_features[sid] = {
            'category': session['category'],
            'embeddings': embs,
            'risk_turns': [t['risk'] for t in client_turns],
            'turn_indices': [t['turn_idx'] for t in client_turns],
        }

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(raw_sessions)} 완료...")

    print(f"\nbal_features 완료: {len(bal_features)}세션")
    return bal_features


def compute_levy_features_window_v2(features_dict, w=20, s=10,
                                     pre_turns=20, baseline_turns=30):
    """
    Canonical, bug-free feature extraction (paper §3.4-3.5).

    For each session: fits PCA once on the full client-turn
    embedding sequence, computes a session-level sequence of
    window-wise Lévy areas (width w, stride s), z-scores that
    sequence within-session, then averages it over the pre-event
    window (the `pre_turns` turns immediately preceding the first
    eligible risk turn) and the baseline window (the first
    `baseline_turns` turns of the session).

    Unlike the earlier buggy pipeline, this recomputes windows
    directly from the full sequence for every session rather than
    indexing into a fixed-length pre-computed array, so no session
    is silently dropped due to an out-of-range index. On the
    corrected Korean corpus this yields N=71 (vs. the previously
    reported N=70; see module docstring).
    """
    results = []
    for sid, data in features_dict.items():
        emb = np.array(data['embeddings'])
        risk = data['risk_turns']
        if len(emb) < w:
            continue

        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(emb)

        la_seq, la_turn_start = [], []
        for start in range(0, len(coords_2d) - w + 1, s):
            la_seq.append(compute_levy_area_2d(coords_2d[start:start + w]))
            la_turn_start.append(start)
        if len(la_seq) < 4:
            continue

        la_seq = np.array(la_seq)
        if np.std(la_seq) > 1e-10:
            la_seq = (la_seq - np.mean(la_seq)) / np.std(la_seq)

        risk_indices = [i for i, r in enumerate(risk) if r == 1]
        if not risk_indices:
            continue
        # Collapse consecutive risk-flagged turns into their first
        # onset ("risk peaks").
        risk_peaks, prev = [], -2
        for idx in risk_indices:
            if idx > prev + 1:
                risk_peaks.append(idx)
            prev = idx

        for peak in risk_peaks:
            if peak < pre_turns:
                continue
            pre_mask = [(t >= peak - pre_turns) and (t < peak)
                        for t in la_turn_start]
            base_mask = [t < baseline_turns for t in la_turn_start]
            if sum(pre_mask) == 0 or sum(base_mask) == 0:
                continue
            pre_levy = np.mean(la_seq[pre_mask])
            baseline_levy = np.mean(la_seq[base_mask])
            results.append({
                'dialogue_id': sid,
                'baseline_levy': baseline_levy,
                'pre_risk_levy': pre_levy,
                'diff': pre_levy - baseline_levy,
            })
            break  # first eligible risk peak per session

    return results


# ============================================================
# SECTION 2 — Supreme Court oral arguments (ConvoKit),
#              advocate-only filtering
# ============================================================

def filter_valid_sessions_advocate_only(supreme, w=10, s=5, pre_turns=20):
    """
    Anchor definition (paper §3.2, corrected pipeline): within the
    advocate-only utterance sequence for a case (justice
    interjections removed), the anchor is the first turn at which a
    respondent-side advocate (side == 0) speaks, requiring at least
    10 preceding advocate turns.

    This differs from filtering on the full transcript (which
    yields N=1,507 identifiable rebuttal onsets); restricting to
    advocate-only utterances and re-detecting the anchor within
    that reduced sequence yields 4,941 anchorable sessions, of
    which a fixed subset satisfies window-eligibility (paper Table
    3: N=1,998 for the matched pseudo-anchor test; a separate
    500-session subsample is used for the pairing-permutation
    test).
    """
    convs = list(supreme.conversations.values())
    valid_sessions = []

    for conv in convs:
        udf = conv.get_utterances_dataframe()
        speakers = udf['speaker'].tolist()
        texts = udf['text'].fillna('').tolist()
        speaker_types = udf['meta.speaker_type'].tolist()  # 'J', 'A', 'U'
        advocates = conv.meta.get('advocates', {})

        pet_names = [n for n, info in advocates.items() if info.get('side') == 1]
        res_names = [n for n, info in advocates.items() if info.get('side') == 0]
        if not pet_names or not res_names:
            continue

        adv_indices = [i for i, st in enumerate(speaker_types) if st == 'A']
        if len(adv_indices) < w:
            continue

        adv_speakers = [speakers[i] for i in adv_indices]
        adv_texts = [texts[i] for i in adv_indices]

        anchor = None
        for i, sp in enumerate(adv_speakers):
            if sp in res_names and i >= 10:
                anchor = i
                break
        if anchor is None or anchor < pre_turns:
            continue

        valid_sessions.append({'texts': adv_texts, 'anchor': anchor})

    return valid_sessions


# ============================================================
# CLI entry point
# ============================================================

if __name__ == '__main__':
    BASE = os.environ.get('LEVY_DATA_DIR', 'data/')

    print("=" * 60)
    print("Korean counseling corpus")
    print("=" * 60)
    bal_features = build_korean_bal_features(
        base_dir=os.path.join(BASE, 'korean_raw'))
    with open(os.path.join(BASE, 'bal_features.pkl'), 'wb') as f:
        pickle.dump(bal_features, f)

    w20 = compute_levy_features_window_v2(bal_features, w=20, s=10,
                                           pre_turns=20, baseline_turns=30)
    print(f"\n최종 Korean 코퍼스 N = {len(w20)} (기대값: 71)")
    assert len(w20) == 71, (
        f"Expected N=71 (corrected corpus, including dialogue ID A105); "
        f"got N={len(w20)}. Check the raw data and filtering logic above."
    )
    with open(os.path.join(BASE, 'korean_pre_results.pkl'), 'wb') as f:
        pickle.dump(w20, f)

    print("\n" + "=" * 60)
    print("Supreme Court (advocate-only)")
    print("=" * 60)
    print("(requires `pip install convokit`; run separately if not installed)")
    try:
        import convokit
        supreme = convokit.Corpus(filename=convokit.download("supreme-corpus"))
        valid_sessions_adv = filter_valid_sessions_advocate_only(supreme)
        print(f"필터 통과 (advocate-only, anchorable): {len(valid_sessions_adv)}"
              f" (기대값: 4,941)")
        with open(os.path.join(BASE, 'supreme_valid_sessions_adv.pkl'), 'wb') as f:
            pickle.dump(valid_sessions_adv, f)
    except ImportError:
        print("convokit not installed -- skipping Supreme Court preprocessing.")
