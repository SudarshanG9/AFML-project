# 10. Evidence, traceability and references

## How to audit a claim

The [claim registry](results/evidence_claims.csv) records each important claim's owner, exact original artifact, branch/commit where available, SHA-256, split, result location, verification status and limitation. The [result-table index](results/README.md) describes the display extracts. Each numerical CSV row includes the original source hash. The CSV is a public traceability index, not a copy of private media, labels, predictions or input archives.

For final numerical claims, start with original Member 2 `test/metrics.json` SHA-256 `fae7f7f8ed9f302ed4a298f8ee66f00d3b59ba679ff2f4424aad054f26355080`. Its `configs` contains all overall and per-type metric blocks; `paired` has four overall contrasts; `paired_by_type` has the twelve category contrasts. The source is the nested original `m2_final_test_package.zip` in the supplied small Drive ZIP. All eleven checksum-listed payloads match. The output records split test, seed 0, n_boot 1000, fixed threshold and manifest hash. The source commit printed by the original log is `70e3a57`; its full repository identity is `70e3a57bf6716aa39a6fe12776fe235f802d0662`.

The frozen C9 registry is [frozen_precision_policy.json](../configs/dayC/member1/frozen_precision_policy.json), SHA-256 `4951455481405602e835902da04478afa10eb9707935e6208aed6a813a0401bf`. It binds checkpoint, canonical cohort, original validation metrics/paired results, modes, release acknowledgements and source files. Many original runtime artifacts are intentionally external/ignored; a path in the registry is an original locator, not a promise that private bytes are in a clean checkout. Verify them against the owner-supplied original package. Do not replace a missing artifact with a newly computed one under the old name.

## Principal identity anchors

| Evidence | Identity and location | Verification and qualification |
| --- | --- | --- |
| Pretrained LAV-DF DFD checkpoint | `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e` | accepted identity; frozen weights; not published by cleanup |
| Canonical whole-video manifest | `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd` | supplied original bytes match; private cohort labels not republished |
| Calibration protocol | `358c037242f6980fed2a8d0ff0ef830f17b4789676b391c8f6dff245842b37dd` | reference validation; threshold fixed before final test scoring |
| Accepted public Member 1 return | `97e603cb3883023ba87cb2fec518e69ce8a5a3438cf79827db72be56bbee4f0c` | successful R2 outputs, output inventory and earlier attempt ledgers |
| Accepted private recovery return | `4b452b2eadb280604f9daaf606300fef95b78b8f63d1280412cb250e19e97e60` | updated Phase 1 accepted hash; additional private provenance remains external |
| Final evaluation metrics | `fae7f7f8ed9f302ed4a298f8ee66f00d3b59ba679ff2f4424aad054f26355080` | primary saved held-out numbers; eleven-file package inventory passes |
| Final generated summary | `2b6fdabd78dd76594298bc4deabdba8faacec1e8284265b03e5d6404601070ce` | documentary companion, not substitute for unrounded values |
| Test-access ledger | `adc137a2bd040f759dc23389cd86e4a1c1716290c582539e712b117742843719` | two final entries; times have no timezone offset |
| Silence-control JSON | `5fe6b0530998ef36a2c222bf58065b03214195f5dae1106f67124cfbc86eda69` | reference-only matched 912 subset |
| CPU short/typical/long files | `191429abd9005adda2c23babdad5781f3589d2031bf39604c9540143cf29cfd3` / `32395daa1da7b4ff3cf5873bfdc37962895797d23a60d732db491107426d17d4` / `77b8feff30b52e7db68e680c7bec720c472344291ceee3ff086ba5a1b28539a1` | twelve original rows, 600 samples; linear percentile arithmetic matches |

The three runtime files originate at `artifacts/dayC/member1/day8/B2_RETURN_AUDIT_20261112T160215000947Z/received/benchmarks/four_modes_{short,typical,long}.jsonl` and are duplicated in the supplied Member 3 evidence handoff. Original per-row Git revision is null; accepted release/source inventories supplement it. This is not repaired by inventing a revision inside an immutable result.

The original figures in [Chapter 6](06_EXPERIMENTS_AND_RESULTS.md) retain SHA-256 `562ba294f92d0183bffa628641817b5d9f16592370c91cc57cee4c932ff1098c` (curves) and `347640ba3c8218110b3333e5f14f0b1103f681ecdd1f918e81767fcdcaac6afc` (recall). Their package bytes equal the versions already published on main. No figure or detector statistic was regenerated during cleanup.

## Reference inputs, chronology and superseded reports

The 18-page `AFML Final.pdf` has SHA-256 `12df449ef6239f425670070e839d371e3e413822f328271c8d1ba624e640a9fa`. Text from every page was inspected, with rendered page 1 and page 18 checked for the question/hypothesis registry. The supplied Elicit critical assessment hash is `7d6a36dcd8635926ef27595a09e0e43a27c5c0ef80d10082562c01f2f97b7dad`; its eight-row comparison CSV hash is `a8601b2143f419d3087311c76a862ca5bfc72d0c0f09194e2a9ac125586b641a`. These private research references remain outside Git. Original writing here synthesizes their reasoning with actual code/evidence rather than republishing their prose as the final report.

The local completion bundle dated 14 November and initial Phase 1 audit are derivative research locators. Their statements that final Member 2 originals were unavailable were superseded by `AFML_Evidence_Verification_20261115T085131Z_UPDATED`, whose verdict is CONDITIONAL_GO with scoped scientific qualifications. That later audit directly verified the supplied accessible evidence copy. It does not certify an inaccessible Downloads staging directory as byte-identical. These local audit files are not published as new primary experimental data.

The source branch tips at cleanup intake are recorded in [SOURCE_BRANCHES.json](cleanup/SOURCE_BRANCHES.json). Current main is `a6c9bf786efe5e5290baf9ec24d5efb7d12fda80`; Member 1 is `9eb32dec1ecf9a0c901b520bcd7252b8ecb60a8b`; final Member 2 branch is `b852bcc48e9627909c3b1139148d95e48352fe5b`; later integrated runtime comes from backend paths of `6de021adb93142c3f3a2383c7ae8f834101157c0`. The common cleanup foundation records their explicit tree composition without a Git merge. The current reporting/CLI extensions must not be presented as byte-identical to the historical C9 source bundle.

## Verification limits that remain

The exact final evaluation shell invocation is not printed in its original console log. The README's example invocation is useful documentation but is not a historical command transcript. The original cache's complete per-feature byte baseline and independent original pairwise tensor reproduction remain unverified; the saved full-array check explicitly reached only 2,800/3,200. Raw media, full feature caches and the full large metadata population were not re-audited for this cleanup. No held-out model inference, evaluator, training, new benchmark or test-label-derived metric was run.

A separate original Elicit pretrained-model assessment, older engineering/literature PDFs referenced inside Elicit, and an original Undermind response were not found among supplied project files/archives. Their supposed details are not invented. Primary AuthentiQ full text was not accessible; its specific separate-model interpretation is marked secondary. The AV-DeepFake-Net CSV DOI differs from the inspected institutional citation; the direct named-work record below is used. Approvals/acknowledgements remain attestations rather than signatures, including the disclosed Member 3 nested actor inconsistency.

## Bibliography and verification scope

1. **Project lead.** AVFuse-CausalQ — 12-Day Execution Plan, private supplied `AFML Final.pdf`, 18 pages. Original research question, scope and hypothesis registry; not a publication or proof of execution.
2. **Oorloff et al. (2024).** [AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://arxiv.org/html/2406.02951v1), CVPR. Primary full text inspected for two-stage representation/fusion methodology.
3. **Koutlis and Papadopoulos (2024; revised 2025).** [DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization](https://arxiv.org/html/2411.10193v2). Primary full text and [official code](https://github.com/mever-team/dimodif); actual architecture additionally verified from selected source/checkpoint config.
4. **Khalid et al. (2021).** [FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/d9d4f495e875a2e075a1a4a6e1b9770f-Abstract-round2.html), NeurIPS Datasets and Benchmarks. Primary proceedings; not the final evaluated dataset here.
5. **Cai et al. (2022/2023).** Do You Really Mean That? / Glitch in the Matrix: content-driven audiovisual forgery detection/localization and LAV-DF. [Official paper, data and code repository](https://github.com/ControlNet/LAV-DF), including access terms and license.
6. **Dao et al. (2022).** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS. Primary paper record for IO-aware exact attention.
7. **Zhang et al. (2025).** [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367), ICLR. Primary paper; distinct from our quantizer/backend.
8. **Oh, Kwon and Lee (2026).** [QFlash: Bridging Quantization and Memory Efficiency in Vision Transformer Attention](https://ijcai.org/proceedings/2026/167), IJCAI-ECAI; [official Triton implementation](https://github.com/EfficientCompLab/qflash). Not the separate FPGA work sharing the name.
9. **Li et al. (2026).** [DeFakeQ: Enabling Real-Time Deepfake Detection on Edge Devices via Adaptive Bidirectional Quantization](https://arxiv.org/html/2604.08847v1). Primary full text for HAQ/VEFT methodology; not reproduced here.
10. **Tao et al. (2025).** [Unlocking the Potential of Lightweight Quantized Models for Deepfake Detection](https://ijcai.org/proceedings/2025/59), IJCAI, pp. 520–528. Primary proceedings for connected quantized blocks and shifted logarithmic quantization.
11. **Iyengar, Mallampalli and Sanghvi (2026).** [AuthentiQ: Real-Time On-Device Deepfake Detection for the Internet of Multimedia Things](https://doi.org/10.1109/MIPR70517.2026.00090), MIPR. Identified through supplied Elicit sources; primary full text inaccessible in this audit, so detailed implementation claims remain unverified.
12. **Saleem et al. (2026).** [AV-DeepFake-Net: Attention-Guided and Uncertainty-Aware Network for Audiovisual DeepFake Detection](https://napier-repository.worktribe.com/output/5235118/av-deepfake-net-attention-guided-and-uncertainty-aware-network-for-audiovisual-deepfake-detection), IJIMAI. Institutional primary publication record inspected; conflicting CSV DOI not silently reconciled.
13. **Waseem et al. (2026).** [Sensors-Driven Multimodal Deepfake Detection: A Cross-Attention Fusion Approach with Adaptive Modality Gating](https://www.mdpi.com/1424-8220/26/12/3695), Sensors 26(12), 3695. Primary publisher article; secondary assessment's venue label corrected.
14. **Koutlis and Papadopoulos (2025/2026).** [AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization](https://arxiv.org/abs/2511.18993), arXiv/WACV. Primary abstract/author code identified; not integrated as another experimental model.
15. **Cai et al. (2024).** [Explicit Correlation Learning for Generalizable Cross-Modal Deepfake Detection](https://arxiv.org/abs/2404.19171), ICME. Primary paper for content-level correlation and synchronization caveats.
16. **Smeu et al. (CVPR 2025; preprint 2024).** [Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning](https://arxiv.org/abs/2412.00175). Primary shortcut study; distinct from DeepFake Doctor.
17. **Klemt, Segna and Rohrbach (2025).** [DeepFake Doctor: Diagnosing and Treating Audio-Video Fake Detection](https://arxiv.org/abs/2506.05851). Primary paper record for dataset/method/protocol concerns.
18. **Jain and Wallace (2019).** [Attention is not Explanation](https://aclanthology.org/N19-1357/), NAACL, pp. 3543–3556. Primary proceedings; motivates testing outcomes rather than equating maps with explanations.
19. **Vaswani et al. (2017).** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS. Transformer attention background.

Publication claims and project results have different sources. A paper establishes the authors' method/results in their setting; the accepted original packages establish what AVFuse-CausalQ observed. No cross-paper leaderboard comparison or global novelty proof is inferred from this bibliography.
