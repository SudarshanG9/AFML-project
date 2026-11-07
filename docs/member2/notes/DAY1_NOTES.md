# Day 1 notes: dataset card and label taxonomy

These notes explain what the Day 1 work is, why it comes first, and the decisions in it.
Files covered: `docs/member2/DATASET.md`, `src/avfuse/member2_data/labels.py`,
`tests/member2/test_labels.py`.

---

## 1. What problem Day 1 solves

The whole project asks one question: **when the detector's fusion step is run at lower
precision, does it get worse at catching deepfakes, and for which kinds of deepfake?**

That question can only be answered against **ground truth**: for every clip, we must know
for certain whether it is real or fake, and *what* was faked. If the labels are wrong or
inconsistent, every metric later (AUROC, recall, confidence intervals) is wrong too, no matter
how good the model or the code is.

So Day 1 is not about code that runs fast. It is about agreeing, in writing, on:

1. **Which data** we use and whether we are allowed to use it (licence).
2. **What each label means**, in exactly the words the rest of the team uses.
3. **How clips are grouped**, so that later splits and confidence intervals are honest.

The team plan's Day 1 gate is "viable model + data". This work is Member 2's half of that gate.

---

## 2. The dataset: LAV-DF

- **What it is:** LAV-DF (Localized Audio-Visual DeepFake), built from VoxCeleb2 interview videos.
- **Licence:** CC BY-NC 4.0, research and non-commercial use only. The clips show real people,
  so no video, audio, frame or feature file is ever committed to git.
- **Why LAV-DF and not FakeAVCeleb:** FakeAVCeleb access did not arrive in time. The plan says one
  primary dataset is enough ("second-dataset transfer is optional"), so FakeAVCeleb is now an
  optional second dataset, not a blocker.

### What makes LAV-DF different

A LAV-DF fake is **not** a fully fake video. Only **a few words** are replaced inside an
otherwise real video. The replaced part can be:

- the **audio** (new words spoken in the person's voice),
- the **video** (lips changed to match new words),
- or **both**.

The metadata lists where the edit is (`fake_periods`, start and end times).

**Consequence:** if we cut a 2-second window from a fake video at a random place, it may contain
no edit at all, so a "fake" label would sit on real content. That is why fake clips are cut
**centred on the edit**, and a fake whose edit falls outside the model's window is excluded
(Days 3-4 implement this; Day 1 records the rule).

---

## 3. The label taxonomy (what `labels.py` does)

### The problem

Every dataset describes fakes in its own way:

- LAV-DF uses two yes/no flags: `modify_video` and `modify_audio`.
- FakeAVCeleb uses folder names like `FakeVideo-RealAudio`.
- The team contract (`contracts/schemas/manifest.schema.json`) uses **one** field,
  `manipulation_type`, with exactly four allowed values.

If each member translated these on their own, small differences (for example `fake_audio` vs
`audio_fake`) would silently break the join between labels and predictions.

### The mapping

| `modify_video` | `modify_audio` | `manipulation_type` | `label` |
| --- | --- | --- | --- |
| 0 | 0 | `real` | 0 |
| 0 | 1 | `audio_fake` | 1 |
| 1 | 0 | `video_fake` | 1 |
| 1 | 1 | `both_fake` | 1 |

Rule: **a clip is fake (label 1) if either stream is manipulated.**

### The functions in `labels.py`

- `BY_FLAGS`: the table above as a dictionary, `(video_fake, audio_fake) -> manipulation_type`.
- `BY_CATEGORY`: the same four types for FakeAVCeleb-style folder names, so the optional second
  dataset maps to the same words.
- `manipulation_type(video_fake, audio_fake)`: looks up the type. It accepts `True`/`False` or
  `1`/`0`, because LAV-DF's metadata stores booleans.
- `label_of(mtype)`: returns 0 for `real`, 1 for any fake, and **raises an error** for an unknown
  name. A typo such as `fake_audio` fails loudly instead of quietly becoming a wrong label.
- `FAKE_TYPES`: the three fake types, built from the shared constants in
  `src/avfuse/shared/contracts.py` instead of being typed again, so there is a single source of
  truth.

### Why four types instead of just real/fake

The research question is **per attack type**. Quantization might hurt detection of audio fakes
while leaving video fakes untouched. A single real/fake label would hide that. Each fake type is
later compared **against the real clips** (a fake-only subset cannot produce a valid AUROC,
because AUROC needs both classes).

---

## 4. Grouping decisions (in `DATASET.md`)

### `source_id`

The original video a clip came from. A fake and the real video it was edited from share the
same `source_id`.

**Why it matters:** a fake and its own original look almost identical. If one is in training and
the other in test, the model can "recognise" the test clip instead of genuinely detecting a fake,
and the test score is inflated. So everything with the same `source_id` must stay in one split
(checked on Day 2).

### `identity_id` (a stated limitation)

The team contract asks for a person identity as well. **LAV-DF's metadata has no speaker ID.**
So `identity_id` is set equal to `source_id`, and every manifest row says so in its `notes` field.

Consequence, stated honestly: splits are **source-disjoint**, but not guaranteed
**speaker-disjoint**. The same speaker can appear in two splits through two different source
videos.

---

## 5. Sample sizes

| Split | Clips per category | Categories | Target |
| --- | --- | --- | --- |
| train | 1,500 | 4 | 6,000 |
| val | 300 | 4 | 1,200 |
| test | 500 | 4 | 2,000 |

Measured on the test split: 2,000 sampled, 1 too short, 7 fakes with the edit outside the
window, so **1,992 usable** (500 real, 499 audio_fake, 493 video_fake, 500 both_fake).

Balanced categories matter because per-type metrics need enough fakes of each type; otherwise
one attack type would have wide, uninformative confidence intervals.

---

## 6. The tests (`test_labels.py`)

| Test | What it proves |
| --- | --- |
| `test_flags_map_to_shared_types` | each flag pair maps to the right type, including `True`/`False` input |
| `test_labels_follow_types` | `real` is 0, every fake type is 1, and a wrong name like `fake_audio` raises an error |
| `test_legacy_categories_cover_all_types` | the FakeAVCeleb names cover exactly the same four types |

Run with: `pytest -q tests/member2/test_labels.py`

---

## 7. Likely questions and answers

**Why is a clip with only fake audio labelled "fake"?**
Because the question is "has this clip been manipulated?". Any manipulated stream makes the
whole clip untrustworthy. The type (`audio_fake`) keeps the detail for per-attack analysis.

**Why not let each member define their own label names?**
Labels from the manifest are joined to predictions by `clip_id` and compared by
`manipulation_type`. One spelling difference breaks the join or silently drops a class. One shared
mapping, tested, prevents that.

**What is the biggest limitation of the Day 1 decisions?**
No speaker IDs in LAV-DF, so splits are source-disjoint but not speaker-disjoint. It is written
down rather than hidden.

**Why does Day 1 come before any metric code?**
Every metric is computed against these labels. Wrong labels make every later number wrong,
however correct the code is.

---

## 8. How Day 1 connects to the next days

- **Day 2** uses `source_id` / `identity_id` to check that no group crosses train/val/test.
- **Days 3-4** apply the "cut fakes around the edit" rule recorded here.
- **Days 5-6** compute metrics per `manipulation_type`, each fake type against all real clips.
