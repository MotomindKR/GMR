# Mini default: validation handoff

Selected 2026-09-15: `body-arm-half-w40` with the yaw-stability correction below,
now the sole active Mini JSON.
Configuration SHA-256:
`6b26709170ce9469207ce7f46bea3cd801988ed13d0cc1536e5ba5b9cd79516f`.
Upstream GMR: `bb1bbe40774794fceb2a7c579a3464a28e68c844`.

The correction changes only three weight entries: stage-1 torso orientation
cost becomes `[25, 25, 10]`; both stage-2 foot orientation costs become
`[50, 50, 10]`. This gives both stages a torso-yaw objective while relaxing
foot yaw, not foot tilt. Scales, anatomical offsets, all other costs, geometry,
initialization, and upstream solver behavior are unchanged. No filter is added.

## Stability correction

Full-clip comparisons use the same source frames and initialization. Jitter
is joint-angle second-difference RMS at 30 Hz, excluding the first 0.5 seconds
for both versions; tracking P95 includes startup and pools both wrists/feet.
This is not a measured motor-speed or hardware safety limit.
Regression checks cover six clips / 3,247 frames: bottle pour, cube pick,
deep lunge, sit/stand, military crawl, and the existing 30-second yoga window.
Wrist XYZ/orientation P95 and waist/leg jitter RMS improve in all six. The full
historical 56-clip corpus has not been rerun after this correction.

| Metric | Previous default | Revised default |
|---|---:|---:|
| Bottle pour waist jitter | 3.600 deg | 0.170 deg |
| Sit/stand leg jitter | 3.001 deg | 1.663 deg |
| Sit/stand peak leg second difference | 30.23 deg | 19.92 deg |
| Bottle pour wrist XYZ / orientation P95 | 8.41 cm / 3.11 deg | 6.70 cm / 2.63 deg |
| Sit/stand wrist XYZ / orientation P95 | 9.38 cm / 2.02 deg | 8.90 cm / 1.86 deg |
| Sit/stand foot XYZ / orientation P95 | 3.03 cm / 2.76 deg | 1.72 cm / 12.14 deg |

This reduces twitching, not all motion irregularity. The sit/stand source has
large foot-yaw changes; reduced yaw accuracy is the explicit tradeoff. Crawl
remains unsuitable for execution: its maximum leg second difference worsens
from 32.05 to 37.27 deg, despite reduced RMS jitter, improved wrist tracking,
and lower maximum collision penetration (15.96 to 12.83 cm). No unrestricted
deployment approval follows from this change. Yoga also has a larger isolated
knee step at clip time 26.7 seconds (peak leg step 14.38 to 25.22 deg), despite
lower RMS jitter; peak leg second difference increases from 9.39 to 19.19 deg.

## Historical calibration evidence (before the stability correction)

Evaluated 56 clips / 15,341 frames at 30 Hz, retaining startup failures:
20 manipulation, 22 complex, eight previously inspected additional-subject
manipulation clips, and six fresh clips selected before final validation.
Fresh clips: GRAB s5 cup pour, cube inspection, binocular use; ACCAD Male2
forward crawl, backward crawl, and lie-to-crouch. This is kinematic validation,
not closed-loop simulation, contact-ground-truth evaluation, or commissioning.

Median per-clip P95 errors on the original manipulation set:

| Metric | Previous Mini | Selected Mini | Stock G1 |
|---|---:|---:|---:|
| Wrist XYZ | 10.17 cm | 7.15 cm | 9.55 cm |
| Wrist orientation | 1.66 deg | 1.91 deg | 1.17 deg |
| Inter-hand XYZ | 11.59 cm | 8.31 cm | 6.85 cm |
| Inter-hand orientation | 1.92 deg | 2.34 deg | 1.21 deg |
| Torso orientation | 12.47 deg | 12.25 deg | 3.09 deg |
| Joint second-difference RMS | 1.10 deg | 1.24 deg | 0.41 deg |

Errors are independently measured, not weighted solver residuals. Each robot
has its own anatomical targets and collision geometry. Against the unchanged
previous Mini wrist targets, the selected profile scores 9.21 cm, rather than
7.15 cm: calibration changes explain part of the apparent gain. Held-out
articulated-reference arm landmark RMS improves from 8.17 to 5.99 cm.

## Not approved for unrestricted physical execution

- Original 50 clips: three self-overlap flags above 3 cm, unchanged in count.
  Worst overlap: yoga 5.00 cm, sit/stand 8.46 cm, lie-to-crouch 3.21 cm.
- Military crawl still reaches 15.96 cm floor penetration. All three fresh
  complex clips also have substantial floor failures.
- Crouch-to-run startup floor penetration worsens from 5.03 to 6.84 cm.
- Stress-set motor-coordinate rates reach 48.74 rad/s and 1747.57 rad/s^2
  after startup, including coupled-ankle transmission coordinates. No
  hardware-rate/torque commissioning was performed.
- Source-stationary-foot proxy P95 XY excursion is essentially unchanged:
  2.872 to 2.880 cm across 382 intervals. It is not measured contact truth.

The 3 cm overlap screen is diagnostic, not a safe allowance. Individual motions
still require external runtime validation before training use or deployment.
The cleanup removes experimental diagnostics, not production safety checks.

## Preserved model identity

Generated physics model SHA-256:
`b06ab05c0a5fd58f5feeba22f051a3a68328c1c8188faf194e18814983da4744`.
Generated visual model SHA-256:
`6934af73340a92e9c33aa5bf3926b77bce188e783c984deb852ac94f7978f4b7`.
These identify the generated models used for the offline audit. Project models
are generated from the pinned canonical description; see [integration](BELLO_INTEGRATION.md).

Exploratory helpers, alternate profiles, generated trajectories/reports/media,
and temporary environments were removed from the working area during cleanup.
They were moved to macOS Trash for recovery, not permanently erased. The NAS
source data, production dependency pins, upstream tools/assets, and unrelated
work were not changed. The separate NAS-cleanup receipt is retained.
