# Competitive landscape — RSNA Knee Abnormality Detection

Scraped 2026-09-14 from Kaggle's public APIs (leaderboard, 109 forum threads, 300 public
notebooks) plus the downloadable source of the public 0.941 notebook. Re-run before relying on
the numbers; the field moves.

> **Sourcing caveat.** Kaggle gates discussion *bodies* (403 `forumTopics.get` for anonymous
> sessions, unindexed by search engines). Thread-level evidence here is titles, authorship, votes
> and comment counts only. Notebook *source* is downloadable without login and is where the real
> detail came from. A logged-in pass over the threads named below is still worth doing.

## Where the bar is

| | score | rank |
|---|---|---|
| 1st | 0.956 | — |
| **gold cut** | **0.951** | **17** (gold = 10 + teams/500; ~20 at 5k teams) |
| silver cut | 0.941 | 186 |
| bronze cut | 0.941 | 372 |
| **best public notebook** | **0.9410** | — |
| plateaus | 0.941 (352 teams), 0.936 (354 teams) | — |
| field | 3,723 teams, 4,143 competitors, 45,729 submissions, median 5 subs | — |

Gold is **+0.010 over a notebook anyone can fork**, and ~350 teams are already on it.

## The top 30 publish nothing

All solo teams; median 73 submissions. Full public footprint of the gold zone:

| rk | score | team | public footprint |
|---|---|---|---|
| 3 | 0.955 | Mazzutti | **authored** "Is the gated KneeCoT dataset permitted as external data?" (#734109); **replied** on "Rules clarification: external knee-MRI datasets, and using an LLM API to derive labels from the reports" (#733652) |
| 8 | 0.953 | Matteo Vitali | **authored** "Public/private test split — stratified by site, or entire sites held out?" (#734681) |
| 13 | 0.953 | Tucker Arrants | **replied** on "Plateau at 0.903" (#740610), "Too many noise public notebooks" (#736268) |
| 34 | 0.949 | Jiwei Liu | the only published notebook in the top 60: "RSNA Knee Fast 2xT4 Inference" (0.941) |

The other 26 of the top 30 have zero threads, zero replies, zero notebooks. The two questions they
did ask are about **external supervision** and **split structure** — not backbones.

**The public stack's own authors are not in the gold zone:** pilkwang (568-vote baseline) rank
1263 @ 0.935 · mattiaangeli 135 @ 0.942 · renta.k 158 @ 0.941 · dreaddevelopment 176 @ 0.941 ·
tonylica 341 @ 0.941 · prvsiyan 553 @ 0.940. They top out at ~0.942. The remaining +0.010 is
private and cannot be forked.

Informative threads from outside the top 30 (read these logged-in): stevenleehans' "'Not
addressed' is a label too — what we learned reading 4,407 knee reports with an LLM" (41 votes) and
"Scaling the encoder bought us nothing (+0.0011)" (14) · tonylica's "What Could the Final Ceiling
Be" (15) · dreaddevelopment's "RSNA Raptor Weights" (16) · "Why everyone's Synovitis AUC is stuck
around 0.6-0.7 (it's not your model)" · "Possible inconsistencies between MRI reports and provided
labels" (25).

## The public 0.941 pipeline, in full

From the rank-34 notebook's source and its preserved outputs.

| stage | models | blend role |
|---|---|---|
| 1a | multilingual **rule-based** report-label extractor | built the weak supervision |
| 1b | **20 DINOv2 members** (12 blocks, last 6 trainable, 10.7M params, dim 768), bit-identical frozen 6-block prefix + per-member tails | base ranking |
| 2 | **5 A5 attention-pooling folds** (`vit_small_patch16_dinov3.lvd1689m`, pool=xcodex) | 45% against stage 1 |
| 3 | **RadImageNet ResNet-50**, E10/E13/E11 heads | reference, second pass, calibration |
| 4 | **4 Raptor CoAtNet views + residual CoAtNet** | final per-finding routing |

Fusion is a **weighted rank mean**. 41 checkpoints total, all one family.

Preprocessing (solved, and better than a week-1 rebuild):
- **6 anatomical slots**: `SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1`
- cache `(studies, 6, N, 336, 336) uint8`; models run at 224 and 336
- **crop by physical mm (130 mm), not pixels** — their EDA: a 140 mm crop spans 398–896 px across
  series, a 2.3× range
- laterality resolved from the DICOM tag **and** from geometry, then normalised by flip
- **per-target slot priors** (strength 0.55): ACL → sagittal, Baker's → `SAG_FLUID_FS` only,
  PF OA → axial
- TTA with **per-target pooling modes** (Synovitis differs from the rest)

## Numbers from their EDA outputs

- 4,407 training studies · 24,371 training series · ~1,322 test studies
- **58 studies carry all twelve labels**; the rest are supervised only by the report
- **0 studies have no report**
- Slot coverage: Axial fat-sat **100%** · Sagittal no-FS 96.8% · Coronal FS 96.4% · Sagittal FS
  94.2% · **Coronal no-FS 77.3%** · **Axial no-FS 19.4%**
- Label co-occurrence: Medial OA + Baker's **+0.48** · Medial Meniscus + Medial OA +0.42 ·
  Lateral Meniscus + Lateral OA +0.42
- Public label tables vs the 58 gold: LLM **0.893**, regex **0.814**

## The two documented weaknesses

**1. Zero slots.** Their own note: *"A slot no study fills is a column of zeros fed to the encoder;
a study that fills none of them can only be given a constant, and under ROC-AUC a block of
identical constants earns half credit against everything it ties with. That is the cheapest score
leak in this whole pipeline."* They diagnosed it and shipped anyway — ~81% of studies feed zeros
into the Axial-no-FS slot, ~23% into Coronal-no-FS.

**2. LB-fitted fusion.** *"The parent scored 0.939 with 0.60 across the board. Probe #22 raised
five findings, and put Lateral Meniscus at 1.00 — meaning three of the four stages are discarded
entirely for that column. That bought +0.002 on the public split."* Per-target transformer/Raptor
routing: Lateral Meniscus 0.00/1.00 · ACL 0.25/0.75 · Medial Meniscus 0.20/0.80 · most others
0.40/0.60. Fitted on 30% of the test set, no OOF backing, now shared by ~350 teams. With label
columns correlated up to +0.48, per-target weight tuning overfits fast — as their own note says,
*"findings that co-occur are findings an ensemble gets right or wrong together — worth remembering
before reading any per-finding weight as skill."*

## Public assets worth attaching

| asset | uses | what |
|---|---|---|
| `metaresearch/dinov2` (small) | 209 | dominant frozen encoder |
| `pilkwang/rsna-knee-llm-labels` | 111 | LLM-read labels |
| `stevenleehans/rsna-knee-llm-report-labels` | 66 | **0.893 vs gold; regex 0.814** |
| `lixin73/rsna-knee-llm-report-labels-sol56` | 40 | third label table |
| `marwanmath/resnet-50-radimagenet-marwan` | 89 | RadImageNet R50 |
| `dreaddevelopment/knee-raptor-corpus` (+`-ext`) | — | preprocessed fixed-size volume stacks |
| `barun2104/rsna-knee-mri-processed-3d-volumes` | — | 24,371 uint8 3D volumes (.npz) |
| `flight0234/rsna-knee-dual-grouped-folds` | — | folds guarding the **shared-report and shared-scanner** leaks |
| `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` | 43 | "Public Baseline 0.941: 41 checkpoint members in one offline bundle" |
| `dreaddevelopment/raptor-knee-{maxspan,native384,widefov,finespacing,fullspan}` | 67/33/… | CoAtNet heads per slice-sampling strategy |

No 570 GB preprocessing run is needed — the corpora already exist.
