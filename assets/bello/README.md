# Bello asset provenance

These preserved full-size Bello models were generated from bello-stack commit
`58d44103c73889ef973504793a605c3d476cc8b8`, source URDF SHA-256
`d5049dddd236f5c9895dce4a7a22d85ed077ad85bfcbb2b32e3b439c21c066eb`.

`mjcf/bello_full_body_boxes.xml` contains collision geometry; the viewer model
adds visual meshes. The packaged joint ranges include historical GMR-specific
adjustments and must not be treated as the canonical hardware description.
For project use, the bello_mujoco caller registers its model generated from the
pinned bello-stack description instead.

GMR Python is now unmodified upstream. Earlier descriptions of custom collision
constraints, floor corrections, waist tasks, profile overrides, offline
smoothing and quality gates no longer apply; those extensions were removed.
Only the standard upstream fields remain in the Bello IK JSON. This cleanup
does not recalibrate or commission the full-size Bello profile.

See [integration](../../BELLO_INTEGRATION.md). Other robot assets are unchanged.
