# Bello asset provenance

These MJCF files were generated from the Bello full-body source URDF at
`bello_mujoco_wholebody` commit `56ed6795ea066f8e37c4a28b3a4810a259e0f28d`.
The source URDF SHA-256 was
`379cd481d77ed1d5c92cd1d33cb2ba0d33c60b9cc70b9d93490453b2d8ecdaa5`.

Generation used the source asset pipeline with:

```python
AssetPrepConfig(
    asset="full_body",
    collision_mode="box",
    end_effector_sphere_mode="both",
    neighbor_exclusion_depth=2,
    joint_coordinate_offsets=(
        ("left_ankle_pitch_joint", 0.261799),
        ("right_ankle_pitch_joint", 0.261799),
    ),
    fixed_joint_names=(
        "waist_roll_joint",
        "waist_pitch_joint",
        "neck_yaw_joint",
        "head_pitch_joint",
    ),
    visual_mesh_face_budget=5000,
)
```

`bello_full_body_boxes.xml` is the collision model. The viewer model adds
non-colliding visual meshes. Both models retain the source URDF's five foot
collision primitives per side. Bello's anatomical forward direction is local
`+Y`. The exported reference schema contains 25 actuated joints; the neck and
head remain as fixed bodies so their mass and collision geometry are preserved.

The compiled contact matrix excludes all body pairs at kinematic graph distance
one or two. It remains the simulation contact contract. Bello retargeting does
not reuse that matrix as an IK clearance constraint: expanding every simulation
pair with a large proximity margin over-constrains ordinary arms-down poses and
causes the active set to oscillate. The matrix remains available to downstream
clip validation and simulation instead of perturbing every per-frame IK solve.

The SMPL-X arm mapping follows the same two-stage structure as Unitree G1.
Stage one establishes the shoulder, elbow, and wrist orientation branch. Stage
two retains those orientation targets and adds elbow and wrist positions. The
left and right offsets are calibrated independently from the SMPL-X rest
skeleton, Bello link directions, and each arm's positive elbow-flexion axis.
Wrist orientation has a lower weight than G1 because Bello has one wrist degree
of freedom instead of three.

The lower body retains Bello's positional hip, knee, and foot tracking. The
left and right knee orientation offsets are calibrated independently in the
Bello knee-link frames. Their orientation tasks establish the correct bend
plane, while the second-stage knee position cost stays deliberately soft: a
deep human squat can place the SMPL-X knees outside the workspace permitted by
Bello's link lengths and knee travel. Giving that infeasible position priority
causes the solver to saturate opposite hip-yaw limits and cross the legs. Foot
orientation is projected onto the ground normal before solving, and its
axis-specific cost tracks sole tilt while leaving yaw unconstrained. This gives
the sole a coarse horizontal target without asking Bello's approximately
five-degree ankle roll to reproduce arbitrary human foot roll. After IK, a
one-sided root-height correction prevents either sole box from passing below
the configured ground plane; airborne feet are unchanged. The shared solver
otherwise remains unchanged; Bello does not add posture tasks, output filters,
joint-limit overrides, fixed iteration counts, or collision IK.

The source pipeline rebases the ankle-pitch body frames and coordinates by 15
degrees. This makes joint position zero correspond to a horizontal sole box
without changing the physical joint travel or home pose.
