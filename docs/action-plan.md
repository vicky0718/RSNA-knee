# Action plan — RSNA Knee, target gold (0.951)

38 days to the **Oct 22** final submission. Compute: Kaggle free tier (30 GPU-h/wk, 9 h sessions,
no internet at scoring) + laptop RTX 3060 6 GB. Background and evidence: `competitive-landscape.md`.

## Three bets, in order of expected value

1. **Labels.** Only 58 of 4,407 studies carry gold labels; everything else is supervised by the
   report. The best published extraction scores **0.893 vs gold** (regex 0.814) and ~everyone
   shares the same three correlated label tables. Nobody has beaten 0.893 in public.
2. **Honest validation.** The plateau's per-target fusion weights were fitted by public-LB probes
   (+0.002 on a 30% split, no OOF backing) and its CV likely ignores the shared-report and
   shared-scanner leaks. We fit fusion on leak-safe OOF instead.
3. **The zero-slot fix.** The public stack feeds an all-zero slot to the encoder for ~81% of
   studies (Axial no-fat-sat) and ~23% (Coronal no-fat-sat). Its own author calls this
   "the cheapest score leak in this whole pipeline" and shipped it anyway.

Backbone scaling is spent — the field's own thread is titled "Scaling the encoder bought us
nothing (+0.0011)", and the 0.941 bundle is already 41 checkpoints of one family.

## Submission discipline

5/day, 2 final. **Submit something every day from day one.** Two slots/day for real candidates,
one for a fixed control run (detects LB drift), two held in reserve. Log every run in
`experiments.csv`: config hash, OOF, per-target AUC, LB, runtime. Gold-zone teams average ~73
submissions; without the log that is just noise.

## Schedule

### Day 1–2 (Sep 14–15) — get on the board
1. Join, accept rules, **complete identity verification** (blocks submission; entry deadline Oct 15).
2. Fork `jiweiliu/rsna-knee-fast-2xt4-inference` (readable 0.941 edition), run, **submit**.
   No training, none of our GPU quota. → **0.941 by end of Day 2.**
3. Repo scaffolding + `bin/push_kaggle.sh` (ship `src/` as a private utility dataset; never paste
   code into cells). Verify offline DICOM codecs (`pylibjpeg*`, `gdcm`) in the submission image.

**Deliverable: 0.941 on the board and a working push → run → submit loop.**

### Day 3–6 (Sep 16–19) — honest validation
4. `src/knee/folds.py` — folds grouped by near-duplicate report text (hash + fuzzy) **and**
   scanner/site keys from DICOM headers; cross-check `flight0234/rsna-knee-dual-grouped-folds`.
   Report the CV inflation each guard removes.
5. Recover the public checkpoints' fold assignment from their weight manifests (members are
   fold-tagged: `m_f0..m_f4`, "banked ... fold 2"). If their folds match ours, one pass over the
   4,407 studies per arm yields genuine OOF. If not, our OOF is contaminated and we retrain heads
   ourselves — decide explicitly, do not hand-wave.
6. `src/knee/fuse.py` — re-fit per-target fusion on OOF, shrunk toward the global weight (label
   columns correlate up to +0.48, so per-target tuning overfits fast). Submit it.
   **Expect public to dip**; log it as a deliberate private-LB trade.
7. Measure the real rerun cost on ~1,322 studies. Public notebooks' 3–4 minute runtimes are
   commits against the **3-study** example set, not the rerun.

### Day 3–12 (Sep 16–25, parallel) — the main bet
8. Multilingual open-weights extractor (Qwen2.5-7B-Instruct class, Gemma-2-9B as second reader)
   from Kaggle Models on T4x2 — 4,407 short reports, 1–2 GPU-h, free, and rules-safe: no
   Competition Data leaves Kaggle, so the unresolved hosted-LLM-API question never binds us.
9. Per-finding prompts for the hard targets; emit `positive | negative | not_mentioned` +
   confidence + the quoted span so every label is auditable.
10. Calibrate: `not_mentioned` is not 0. Fit per-class state→probability maps on the 58 gold +
    corpus prevalence with `P(pos) > P(unmentioned) > P(neg)` and a floor; carry per-cell
    confidence into the loss; group the three OA compartments.
11. Mine disagreements against the three public tables; hand-read 50.
    **Gate: beat 0.893 on the 58 gold.**

### Day 8–20 (Sep 21 – Oct 3) — convert the bet
12. Retrain the public-style heads on our labels — cheapest test that the bet is live.
13. **Zero-slot fix**: mask-aware pooling + nearest-slot substitution; never emit a constant
    prediction for a study. Isolated and testable on OOF within hours.
14. **Synovitis / PF OA / Lateral Meniscus specialists.** Synovitis sits at ~0.6–0.7 for
    everyone; with a 12-way macro average, moving it to 0.75 alone is **+0.008** — most of the gap.
15. Cache frozen-encoder embeddings once per corpus so head training costs minutes. This is what
    makes 30 GPU-h/week enough and lets the 6 GB laptop train heads locally.

### Day 20–33 (Oct 3–16) — decorrelated arms only
16. A true **3D arm** (MedicalNet/3D CNN over the volume stacks) and a **report-teacher** arm
    (predict the report's text embedding — dense supervision from 4,349 reports instead of 12 bits).
    Add only if arm-to-arm correlation < ~0.97; reject anything more correlated regardless of its
    solo score.
17. 384 px for meniscus/cartilage targets (the public stack caps at 336).
18. **Oct 15: entry, team-merge and notebook-publishing deadline.** Nothing unresolved after it.

### Day 33–38 (Oct 17–22) — endgame
19. Efficiency-Track variant: smallest arm set within ~0.003 AUC of our best.
    Score = `AUC/(Benchmark - maxAUC) + RuntimeSeconds/32400`, so runtime is a real objective.
20. **Two deliberately different finals**: (a) best OOF ensemble; (b) most robust single pipeline,
    least LB-fitted. Never two variants of one fit — prevalence differs across splits by the host's
    own notice, public is only 30%, and ~350 teams share one LB-probed fit.

## Files on the critical path

`src/knee/reports.py` (labels + gold eval) · `src/knee/folds.py` (leak-safe CV) ·
`src/knee/fuse.py` (OOF-fitted fusion) · `src/knee/corpus.py` (readers for the public corpora and
the 6-slot cache) · then `models.py`, `train.py`, `infer.py`, `metrics.py`.

## Verification

- **Labels** — per-finding agreement vs the 58 gold from `reports.py --eval`, head-to-head against
  all three public tables; 50 hand-read disagreements.
- **Folds** — assert no report-duplicate cluster and no scanner ID spans two folds; report the
  inflation each guard removes.
- **Slots** — assert no study receives an all-zero slot and no study yields a constant prediction.
- **Fusion** — OOF-fitted vs probe-22 weights, scored on both OOF and LB, both recorded.
- **Model** — OOF macro AUC + per-target table; arm correlation matrix before every ensemble change.
- **Submission** — full rerun, internet off, `submission.csv`, 12 columns, ~1,322 rows, runtime logged.
- **End to end** — LB within ~0.005 of OOF, or stop and investigate before anything else.

## Decision gate

**Sep 26.** If our label extraction beats 0.893 on gold, press the bet through Phases 4–5. If it
does not, fall back to the zero-slot fix + OOF fusion + specialists and target a robust silver
plus the Efficiency Track. Either way the call is made with 26 days left.
