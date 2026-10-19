"""
CheckReality AI - Generate Expanded 3,500-Video FaceForensics++ C23 Manifest
Dataset Composition:
- 500 Canonical Actor Pairs partitioned strictly by actor identity:
  - 350 Train Pairs (70%)
  - 75 Tune Pairs (15%)
  - 75 Test Pairs (15%)
- Per Pair:
  - 2 Real Videos (original/u.mp4, original/v.mp4)
  - 5 Fake Videos (1 from each manipulation family: Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures)
- Total Videos: 3,500
  - Train: 2,450 videos (700 Real, 350 Deepfakes, 350 Face2Face, 350 FaceSwap, 350 FaceShifter, 350 NeuralTextures)
  - Tune:  525 videos (150 Real, 75 Deepfakes, 75 Face2Face, 75 FaceSwap, 75 FaceShifter, 75 NeuralTextures)
  - Test:  525 videos (150 Real, 75 Deepfakes, 75 Face2Face, 75 FaceSwap, 75 FaceShifter, 75 NeuralTextures)
- Zero Identity Leakage: Strict grouping guarantees no actor in Train appears in Tune or Test.
"""

import os
import random
import pandas as pd

def generate_manifest():
    base = 'Datasets/video/FaceForensics++_C23'
    methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'FaceShifter', 'NeuralTextures']

    # 1. Discover all canonical actor pairs {u, v}
    pairs = []
    seen = set()
    for f in sorted(os.listdir(os.path.join(base, 'Deepfakes'))):
        if f.endswith('.mp4') and '_' in f:
            pts = f.replace('.mp4','').split('_')
            if len(pts) == 2 and pts[0].isdigit() and pts[1].isdigit():
                u, v = pts
                key = tuple(sorted([u, v]))
                if key not in seen:
                    seen.add(key)
                    pairs.append((u, v))

    print(f"Total canonical pairs available: {len(pairs)}")
    random.seed(42)
    random.shuffle(pairs)
    sel_pairs = pairs[:500]

    train_pairs = sel_pairs[:350]
    tune_pairs = sel_pairs[350:425]
    test_pairs = sel_pairs[425:500]

    split_pair_map = [
        ('train', train_pairs),
        ('tune', tune_pairs),
        ('test', test_pairs)
    ]

    records = []
    for split_name, pr_list in split_pair_map:
        for idx, (u, v) in enumerate(pr_list):
            # Real samples
            r1 = os.path.join(base, 'original', f'{u}.mp4')
            r2 = os.path.join(base, 'original', f'{v}.mp4')
            for r, act in [(r1, u), (r2, v)]:
                if not os.path.exists(r):
                    raise FileNotFoundError(f"Missing real video: {r}")
                records.append({
                    'filepath': r.replace('/', '\\'),
                    'label': 'real',
                    'label_id': 0,
                    'split': split_name,
                    'actor_pair': f'{u}_{v}',
                    'method': 'original'
                })

            # Fake samples: All 5 manipulation families
            for m in methods:
                # Alternate direction u_v vs v_u across pairs to ensure balanced directional swaps
                fname = f'{u}_{v}.mp4' if idx % 2 == 0 else f'{v}_{u}.mp4'
                fpath = os.path.join(base, m, fname)
                if not os.path.exists(fpath):
                    fname = f'{v}_{u}.mp4' if idx % 2 == 0 else f'{u}_{v}.mp4'
                    fpath = os.path.join(base, m, fname)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing fake video: {fpath}")

                records.append({
                    'filepath': fpath.replace('/', '\\'),
                    'label': 'fake',
                    'label_id': 1,
                    'split': split_name,
                    'actor_pair': f'{u}_{v}',
                    'method': m
                })

    df = pd.DataFrame(records)
    print(f"\nTotal records created: {len(df)}")
    print("Split breakdown:")
    print(df.groupby(['split', 'label']).size())
    print("\nMethod breakdown in fake class:")
    print(df[df['label'] == 'fake'].groupby(['split', 'method']).size())

    # Verify zero identity leakage
    train_actors = set()
    for ap in df[df['split'] == 'train']['actor_pair']:
        u, v = ap.split('_')
        train_actors.add(u); train_actors.add(v)

    tune_actors = set()
    for ap in df[df['split'] == 'tune']['actor_pair']:
        u, v = ap.split('_')
        tune_actors.add(u); tune_actors.add(v)

    test_actors = set()
    for ap in df[df['split'] == 'test']['actor_pair']:
        u, v = ap.split('_')
        test_actors.add(u); test_actors.add(v)

    overlap_train_tune = train_actors.intersection(tune_actors)
    overlap_train_test = train_actors.intersection(test_actors)
    overlap_tune_test = tune_actors.intersection(test_actors)

    print("\n--- ZERO-LEAKAGE IDENTITY VERIFICATION ---")
    print(f"Unique actors - Train: {len(train_actors)}, Tune: {len(tune_actors)}, Test: {len(test_actors)}")
    print(f"Overlap Train & Tune: {len(overlap_train_tune)}")
    print(f"Overlap Train & Test: {len(overlap_train_test)}")
    print(f"Overlap Tune & Test:  {len(overlap_tune_test)}")
    assert len(overlap_train_tune) == 0 and len(overlap_train_test) == 0 and len(overlap_tune_test) == 0, "IDENTITY LEAKAGE DETECTED!"
    print("ZERO LEAKAGE CONFIRMED: 0 overlapping identities between all splits.")

    out_csv = 'training/manifests/video_deepfake.csv'
    df.to_csv(out_csv, index=False)
    print(f"\nSaved updated manifest to {out_csv}")

if __name__ == '__main__':
    generate_manifest()
