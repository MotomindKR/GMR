# Mini default: validation handoff

Selected 2026-09-15: `body-arm-half-w40`, now the sole active Mini JSON.
Configuration SHA-256:
`08a68ba2e0e902c87424de9f32abfe085522d769334ad30fff6ad94808f56934`.
Upstream GMR: `bb1bbe40774794fceb2a7c579a3464a28e68c844`.

The profile uses conservative arm scale/position-offset recalibration and
stage-2 wrist position cost 40. The corrected foot datum, orientation settings,
robot geometry, and one-time root/arm initialization are unchanged.

## Evidence

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
