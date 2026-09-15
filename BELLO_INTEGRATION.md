# Bello integration with upstream GMR

All tracked Python matches upstream
`bb1bbe40774794fceb2a7c579a3464a28e68c844`. Upstream robot models, scripts and
examples are preserved. Fork-only solver extensions and diagnostic tools are
removed, not relocated into the solver's caller.

The canonical Mini default for this project is
`general_motion_retargeting/ik_configs/smplx_to_bello_mini.json`.
There are no alternate live/offline profiles. Full-size Bello retains its
standard-field JSON, without its former fork-only settings. Its four inactive
stage-1 elbow/wrist tasks now use their existing stage-2 weights, as approved,
so upstream can construct their target offsets. This is compatibility
adaptation, not full-size Bello motion commissioning.

## Project use

Run `bello-run-reference` from bello_mujoco using its Nix/uv environment.
The caller registers the generated canonical model in `ROBOT_XML_DICT` and
the selected JSON in `IK_CONFIG_DICT` before constructing upstream GMR.
Generated models and licensed SMPL-X assets are not stored in this repository.

At sequence initialization the caller uses the first scaled root target and
the joint seed declared in bello_mujoco's `robots/*_motion.yaml`. It then calls
upstream `retarget` once per new source frame, without overriding upstream
iteration count. No settling loops, floor correction, posture normalization,
custom collision/waist tasks, smoothing or special ankle velocity mapping remain.

The caller retains shape/frame, timing and nonfinite-output checks and external
reference rejection. The policy/hardware stack retains its own safety checks
and outage-to-nominal behavior. Optional upstream velocity limiting is disabled
in this integration. Do not execute unchecked kinematic output on hardware.

See [validation](BELLO_VALIDATION.md) for known failures. The default is selected
for project use; that is not approval for unrestricted physical execution.

## Integrity

```sh
git diff --exit-code bb1bbe40774794fceb2a7c579a3464a28e68c844 -- '*.py'
```
