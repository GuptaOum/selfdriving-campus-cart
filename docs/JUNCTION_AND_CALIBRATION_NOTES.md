# Junction steering + calibration — session notes

Working notes from the session that fixed junction steering, split drivable
classes, added per-frame diagnostics, and fixed a calibration check that
could never fail. Companion to `AUTONOMY.md` (architecture) and
`BUILD_STAGES.md` (build order) — this is the "what changed and why" log.

No training happened in any of this. SegFormer runs exactly as downloaded;
everything below is configuration and geometry around it.

## The mental model

```
webcam -> SegFormer (stage 1: WHAT is drivable, pixel by pixel)
       -> geometry in seg_pilot.py (stage 2: WHERE to aim, plain NumPy)
       -> steer + throttle
```

SegFormer outputs one of 35 class ids per pixel (see
`exported_models/segformer_labels.json`). Everything after that — band
centroids, junction bias, corridor width, steering rate — is code written for
this project, not part of SegFormer. OpenCV's job is the boring parts:
resize, color conversion, morphological mask cleanup, drawing the debug
overlay. The steering math itself is plain array math.

## Commits, in order

1. **`2fd1c3e` — junction commands work at open, unforked junctions.**
   `_pick_run` chooses between SEPARATE drivable runs, but an open paved
   plaza gives one continuous run per band, so LEFT/RIGHT/STRAIGHT all
   produced identical steering (+0.010) and the cart drove straight through
   every turn. Fix: `_target_x` aims OFF-CENTRE within the single run,
   offset from the commanded edge by the cart's required clearance. The
   clearance term is the safety bound — the aim point can never sit closer
   to the boundary than the cart fits, and on a corridor only as wide as
   required it collapses back to the centre on its own.
   Verified on the user's own campus junction photo:
   `LEFT -0.495 / STRAIGHT -0.001 / RIGHT +0.494` (was `+0.010` for all three).

2. **`7105450` — drivable classes split into footpath / road profiles.**
   Including `flat-road` + `flat-parkingdriveway` merged the footpath and
   the street beside it into one corridor, pulling steering toward traffic.
   Measured: -0.370 on a sidewalk with a street in view, -0.009 on the same
   walk with no street in view. Default profile is now `footpath`
   (sidewalk, crosswalk, cyclinglane); `--profile road` in
   `export_models.py` keeps the old set for campus interiors with no
   footpath. **Open:** `SEG_CORRIDOR_FRAC` is still 0.28, tuned for the old
   wider mask; a footpath-only mask is narrower, so this needs re-measuring
   against a taped 1 m grid, not fitting to a video.

3. **`4aa9246` — per-frame CSV + steering rate limit.**
   `vision_bench.py --csv` writes one row per frame (steer, frame-to-frame
   delta, surviving band count, drivable fraction, breaker, clear_m,
   seg_ms) so a spike can be traced to what the mask looked like when it
   happened, not just seen in a min/max. Found: every full-lock (+/-1.000)
   command was an ISOLATED single frame, always where the mask had thinned
   to 1-2 surviving bands (mean |steer| 0.524 at 1 band vs 0.194 at 2).
   `SEG_MAX_STEER_RATE = 1.2` caps the RATE of change (per second, not per
   frame, so a loaded Pi doesn't silently tighten it). Result: 9 -> 5
   saturated frames, mean steering unchanged (-0.049 both times, so real
   steering wasn't damped). **Open:** -1.000 still occurs — at ~1.7 FPS an
   update is ~0.6 s, so a 1.2/s cap still allows ~0.70 of movement in one
   step. A confidence-scaled version (allowance scaled by how many bands
   survived) was drafted but is UNTESTED and not committed — needs
   validating on real campus footage before it goes in.

4. **`dd33dfe` — the camera decides WHEN to turn, GPS only decides WHICH way.**
   GPS (a plain lat/lon fix, nothing else — your code computes the turn
   direction from route bearings, not from any GPS heading) fires a
   junction command purely on distance (<=8 m) with several metres of
   error, so it can arrive before a turning is in sight or after it's
   passed. `detect_arms()` compares the nearest band against the farthest
   band that still has a corridor: anything the near band reaches beyond
   the far corridor's edges is ground opening to the side. A turn is now
   obeyed only if that side is actually open, OR the ground is wide the
   whole way out (a plaza has no single "arm" that stands out — refusing
   to turn there would refuse exactly where turning is easiest). This can
   only WITHHOLD a turn, never invent one. Verified on the junction photo:
   both arms found (~107px/108px), steering unchanged.

5. **`189a936` — replaced a calibration check that could never fail.**
   `calibrate_ground_plane.py`'s reprojection error was always ~0.00 cm,
   because 4 correspondences fit a homography EXACTLY regardless of
   whether the markers were clicked right. Verified: a 40px misclick, a
   left/right swap, and a wrong click order all reported 0.00 cm. Replaced
   with (a) geometry checks — farther-forward must appear higher in image,
   further-left must appear further left, the four clicks must form a
   simple quadrilateral, catching every ordering mistake — and (b)
   `--verify X,Y`, a FIFTH marker measured but not used to fit the
   homography, the only honest accuracy number (a 30px misclick geometry
   can't see reads 7.58 cm on the fifth point). Without `--verify` the
   script now says the calibration is UNCHECKED instead of printing a zero.

## Where things stand

**Phase 2 has no open bugs left that footage alone can fix.** Everything
above was tested on downloaded campus footage (UMich, Illinois Tech) and the
user's own junction photo — never on this cart's own camera.

**Blocked on hardware, not on code:**
- Mount the camera at 45 cm, final position, rigid.
- Tape the ground-plane rectangle PLUS a fifth verification cross; run
  `calibrate_ground_plane.py --verify X,Y`.
- Record the cart's own campus footage and bench it with `--homography` —
  the planner, occupancy grid and tracking have never executed, only
  segmentation + geometry have.
- Re-measure `SEG_CORRIDOR_FRAC` against the taped grid for the new
  footpath-only mask.
- Fine-tune `SEG_CORRIDOR_FRAC`, arm-detection threshold, and confirm the
  `footpath` vs `road` profile choice once real campus video is available.

**Separately, `donkey calibrate` still blocks Phase 1** (the DonkeyCar
behavioral-cloning ablation) — unrelated to any of the above, just also
waiting on the physical cart.

Full before/after tables and raw numbers: `D:\dowlaodsD\junction_fix_results.md`
(kept outside the repo as a working log, not project source).
Test footage: `D:\dowlaodsD\campus_footage\`.
