# Bello asset provenance

These MJCF files were generated from the Bello full-body source URDF at
`bello-stack` commit `58d44103c73889ef973504793a605c3d476cc8b8`.
The source URDF SHA-256 was
`d5049dddd236f5c9895dce4a7a22d85ed077ad85bfcbb2b32e3b439c21c066eb`.

Generation used the source asset pipeline with:

```python
motion_tracking_asset_config(
    collision_mode="box",
    output_root=output_root,
)
```

The GMR copy widens the waist-yaw joint range from the source URDF's
approximately +/-28.6 degrees to +/-45 degrees; its position actuator command
range remains at 90 percent of joint travel. Shoulder pitch is also expanded
to 170 degrees forward and 45 degrees backward on both sides. Because the pitch
axes are mirrored, this is `[-170, 45]` degrees on the left and `[-45, 170]`
degrees on the right. The ankle ranges, remaining body frames, and inertial
properties otherwise come from the source description. The existing decimated
visual meshes are retained because the inertial overhaul did not alter the link
geometry.

`bello_full_body_boxes.xml` is the collision model. The viewer model adds
non-colliding visual meshes. Both models retain the source URDF's five foot
collision primitives per side. Bello's anatomical forward direction is local
`+Y`. The exported reference schema contains 25 actuated joints; the neck and
head remain as fixed bodies so their collision geometry is preserved. The
compiled model includes the physical differential-ankle tendon transmissions.

The compiled contact matrix excludes all body pairs at kinematic graph distance
one or two and remains the simulation contact contract. Expanding every such
pair inside IK over-constrains ordinary arms-down poses and makes the active set
oscillate. The Bello profile therefore uses targeted collision avoidance for
the distal arms against the pelvis, hips, torso, waist, and head, plus each
upper arm against the torso and head. Full-model contacts are still measured by
the downstream quality gate, including leg and hip intersections that are not
safe to resolve with a per-frame arm constraint.

The SMPL-X arm mapping is position-led. Stage one uses upper-arm orientation to
select the shoulder branch. Stage two removes that redundant orientation target
and tracks the elbow and terminal-hand positions. It does not force the SMPL-X
elbow frame orientation, but it does constrain all three terminal-frame
orientation components. Bello's elbow-yaw joint provides the forearm-roll
component that a conventional wrist-roll joint would provide, while wrist pitch
and the upstream shoulder joints complete the six-DoF endpoint pose.

The wrist landmark targets the fixed end-effector body rather than the wrist
joint origin; the sphere geometry is another 20 mm distal to that frame. The
corrected local position offsets preserve the previous world-space hand targets
while using exact anatomical frame rotations. Bello uses one universal arm-task
balance for all clips. Stage one retains upper-arm orientation to establish the
branch. Stage two removes that redundant orientation task, tracks elbow position
at cost `30`, and tracks wrist position/orientation at `50` and
`[30, 30, 30]`. Palm roll about the distal tool axis therefore cannot drift,
while the elbow target still discourages dynamic branch changes.
The `live_upper_body` profile is for real-time landmark sources whose bone-axis
rotations do not observe axial twist. It removes the upper-arm orientation task
in stage one and terminal-hand orientation in stage two, while preserving the
elbow and wrist position costs. A stage-two nominal-posture cost of `10` on each
shoulder-yaw and elbow-yaw joint resolves the remaining position-only null
space without materially competing with the position costs of `30` and `50`.
The default `universal` profile remains the offline SMPL-X contract.
The human wrist's local `-Y` palm normal maps to Bello endpoint `+X`; human
distal `+X` on the left and `-X` on the right map to Bello endpoint `-Z`.
The elbow-yaw ranges are mirrored anatomically: left is -120 to +30 degrees and
right is -30 to +120 degrees. This keeps forearm roll reachable on both sides.

Bello has waist yaw but no waist pitch or roll. Root translation and heading
therefore follow the human pelvis, while root pitch and roll follow `spine3`
through the rigid torso frame. This split prevents seated motions from applying
pelvic recline to Bello's entire trunk when the human spine bends forward to
compensate. The torso and root use the same neutral rotation offset, without the
legacy 12-degree torso-pitch bias.

The sphere is attachment-neutral placeholder geometry; its body frame defines
the reusable palm/tool convention. In the URDF default arm pose, local `+X`
points inward toward the torso on both sides and local `-Z` points distally.
The left endpoint frame is aligned with its wrist parent, while the right
endpoint frame is rotated 180 degrees about local `Z`. The corresponding right
IK rotation and position offsets are expressed in that aligned frame, so this
frame correction does not alter the previously retargeted arm motion.

The lower body uses the same orientation-first structure as the H1-2, Hi, and
N1 mappings. Stage one establishes hip and knee orientation and the knee bend
plane. The knee link's local `X` axis runs along the shank, so its axial
orientation component is weighted at 40 percent of the bend-plane components.
This is strong enough to make the robot knee direction follow the human while
remaining below the bend-plane priority: human shank twist is unreliable near a
straight knee and can otherwise drive Bello's hip yaw toward its limit.
Stage two adds modest hip and deliberately soft knee position costs. A
deep human squat can place the SMPL-X knees outside the workspace permitted by
Bello's link lengths and knee travel; giving that infeasible position priority
causes saturation and crossed-leg solutions. Foot orientation is projected
onto the ground normal before solving. Its axis-specific cost gives toe heading
a moderate weight of 3 and sole-tilt components a weight of 10. This prevents
the toe direction from drifting while using the updated approximately 10-degree
ankle-roll travel without asking Bello to reproduce arbitrary human foot roll.
Waist yaw is tracked as the signed planar angle from the human hip
line to the shoulder line. A relative-frame task applies that angle between
`bello_root` and `torso_link`; this prevents human spine pitch or roll and arm
position errors from reversing or exaggerating the robot's one available waist
axis. After IK, a one-sided root-height correction prevents either sole box
from passing below the configured ground plane; airborne feet are unchanged.

Offline behavior is controlled in `smplx_to_bello.json`. Every clip settles the
first target for 20 passes, applies the model velocity limit, and performs three
five-tap hinge smoothing passes together with a five-tap quaternion filter for
floating-root orientation. Smoothing is reduced per frame when it
would move a palm more than 20 mm/5 degrees, an ankle or torso more than
10 mm/3 degrees, or worsen one of the monitored collision pairs beyond the
configured tolerance. The collision guard caps smoothing-induced penetration
at the same 30 mm limit used by the quality gate. Floating-root translation is
never filtered. Interactive,
headless, and dataset entry points all call this shared offline path.

The dataset converter defaults to a single CPU worker, requires an explicit
2 GB free-memory margin, and rejects clips that fail their JSON quality profile.
The gate measures jitter, full-model contact depth, wrist position and
orientation tracking, and foot orientation tracking. Bello has one universal
threshold set; a rejected clip still receives a `.quality.json` report but is
not exported as a training pickle. Every report also lists limited hinge joints
that spend at least 5 percent of the clip within 1 degree of a bound. Each entry
records the joint's configured range, observed range, limiting bound, and frame
percentage. These are candidate bottlenecks rather than automatic proof of
causality; for example, a straight knee legitimately rests at its lower bound,
and a collision-only failure cannot be attributed to range from proximity
alone.

The model preserves the source URDF's physical ankle-pitch coordinates and
range of approximately -30 to +60 degrees. With the rest of the neutral chain
at zero, approximately +15 degrees of ankle pitch places the sole horizontally.
