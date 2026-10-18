# The two cases, recorded on the rig (2026-10-18, RK3588 unit, same camera, same model)

Both are raw 1080p30 recordings of the USB camera while the demo ran live; the result log of the
live run is kept next to each (`logs/`). The replays through the same binary reproduce the pattern.

## Case 1: all three strips on, good and bad batteries

`case1-good-lights3.mkv`, 9 min 13 s. Live: 16 batteries counted, 10 `Positive + Negative`
(good), 6 with a `Miss_Match` terminal (the bad ones placed on purpose). This is the model doing
its job under the lighting it was trained for.

## Case 2: one strip on, camera on auto, same batteries

`case2-fp-lights1-auto.mkv`, 12 min. Live: 48 batteries, 33 `GO`, 15 `Miss_Match` on batteries
that are good: the lighting false positives, interleaved with correct results, as a real line
would see them. Note what did **not** reproduce it: two strips off with the camera still locked
to the day exposure only lowered the confidence (89 % to 77 %) and kept the results right; the
false positives appeared once the camera was on automatic exposure, as the India line runs it.

## What the logs look like

```
2026-10-18 12:06:21 | ShiftA | TOTAL:68 | GO:53 | NG:15 | RESULT:NG | LL:Positive 88.42% | LR:Miss_Match_Positive 88.42% | ...
```
`LL`/`LR` are the image-left and image-right terminal of one battery; a `Miss_Match` on a
battery that is good is the false positive the correction layer has to catch.

Timeline of every switch (strips, camera mode, restarts): `logs/timeline.txt`.
