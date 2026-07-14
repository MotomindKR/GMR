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
    fixed_joint_names=("waist_roll_joint", "waist_pitch_joint"),
    visual_mesh_face_budget=5000,
)
```

`bello_full_body_boxes.xml` is the collision model. The viewer model adds
non-colliding visual meshes. Both models retain the source URDF's five foot
collision primitives per side. Bello's anatomical forward direction is local
`+Y`.

The compiled contact matrix excludes all body pairs at kinematic graph distance
one or two. GMR derives its IK collision pairs from those compiled exclusions
and MuJoCo's contact masks, so the retargeter and training simulation use the
same matrix.

The source pipeline rebases the ankle-pitch body frames and coordinates by 15
degrees. This makes joint position zero correspond to a horizontal sole box
without changing the physical joint travel or home pose.
