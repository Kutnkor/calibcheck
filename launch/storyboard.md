# Demo storyboard (about 90 seconds; terminal plus browser)

All numbers below come from `launch/demo_transcript.txt`, captured from
`calibcheck demo --out examples/demo_output` (seed 7, synthetic). No video was recorded in
this environment. `launch/record_demo.sh` replays the same steps for a terminal recording, and
the still images are in `docs/images/`.

| # | Show | Say (caption) |
|---|---|---|
| 1 | `head -3 examples/demo_output/data/test.csv` | "Each row has an outcome and the model's probability of the positive outcome." |
| 2 | `calibcheck inspect …/test.csv --prob p_model --id row_id --group segment` | "Validate first. Nothing is guessed: the positive label, the probability column and the IDs are explicit." |
| 3 | `docs/images/reliability_raw.png` | "937 test rows got 0.90. About 65% of them were positive (observed 0.653; 95% Wilson 0.622–0.683)." |
| 4 | `calibcheck fit …/calibration.csv --method isotonic …` | "Fit a recalibrator on separate calibration rows. That fit is in-sample, not evidence." |
| 5 | `calibcheck recalibrate --calibration … --test … --bootstrap` | "Judge it on untouched test rows with paired intervals: isotonic Brier −0.0173 [−0.0214, −0.0129]." |
| 6 | `report.html` in a browser (works offline) | "One portable file with findings, caveats, bins, the configuration and the manifest." |
| 7 | `experiments/results/SUMMARY.md` | "Recalibration is not automatically an improvement: on already-calibrated models, isotonic made test log loss worse in 20 of 20 runs." |
