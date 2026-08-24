
import mink
import mujoco as mj
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

BELLO_ANKLE_TENDON_LOGICAL_JOINT = {
    "left_ankle_motor_1": "left_ankle_pitch_joint",
    "left_ankle_motor_2": "left_ankle_roll_joint",
    "right_ankle_motor_1": "right_ankle_pitch_joint",
    "right_ankle_motor_2": "right_ankle_roll_joint",
}

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool | None=False,
        robot_xml_path: str | Path | None = None,
        task_profile: str | None = None,
    ) -> None:

        # load the robot model
        self.tgt_robot = tgt_robot
        self.xml_file = str(
            ROBOT_XML_DICT[tgt_robot] if robot_xml_path is None else robot_xml_path
        )
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        self.task_profile = self.apply_task_profile(ik_config, task_profile)
        self.offline_solver_config = dict(ik_config.get("offline_solver", {}))
        self.quality_thresholds = dict(
            ik_config.get("quality_thresholds", {}).get(
                self.task_profile or "default", {}
            )
        )
        if use_velocity_limit is None:
            use_velocity_limit = bool(
                self.offline_solver_config.get("use_velocity_limit", False)
            )
        self.use_velocity_limit = use_velocity_limit
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground_alignment_axes = {
            body_name: np.asarray(axis, dtype=float)
            for body_name, axis in ik_config.get("ground_alignment_axes", {}).items()
        }
        for body_name, axis in self.ground_alignment_axes.items():
            if body_name not in self.human_scale_table:
                raise ValueError(
                    f"ground-aligned body {body_name!r} is not in human_scale_table"
                )
            norm = np.linalg.norm(axis)
            if axis.shape != (3,) or not np.isfinite(norm) or norm < 1e-8:
                raise ValueError(
                    f"ground-alignment axis for {body_name!r} must be a finite 3-vector"
                )
            self.ground_alignment_axes[body_name] = axis / norm
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        self.ground_clearance_geom_ids = []
        for geom_name in ik_config.get("ground_clearance_geoms", []):
            geom_id = self.model.geom(geom_name).id
            if self.model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_BOX:
                raise ValueError(
                    f"ground-clearance geom {geom_name!r} must be a box"
                )
            self.ground_clearance_geom_ids.append(geom_id)
        self.planar_relative_yaw_config = ik_config.get("planar_relative_yaw_task")
        self.planar_relative_yaw_task = None
        self.planar_relative_yaw_reference = None
        self.planar_relative_yaw_joint_qpos_address = None
        root_body_id = self.model.body(self.robot_root_name).id
        root_joint_id = int(self.model.body_jntadr[root_body_id])
        if self.ground_clearance_geom_ids:
            if (
                root_joint_id < 0
                or self.model.jnt_type[root_joint_id] != mj.mjtJoint.mjJNT_FREE
            ):
                raise ValueError("ground clearance requires a free robot root")
            self.root_height_qpos_address = int(self.model.jnt_qposadr[root_joint_id]) + 2
        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if self.use_velocity_limit:
            actuator_joint_names = self.velocity_limited_joint_names()
            VELOCITY_LIMITS = {k: 3*np.pi for k in actuator_joint_names}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
        self.collision_avoidance_limit = self.make_collision_avoidance_limit(
            ik_config.get("collision_avoidance")
        )
        if self.collision_avoidance_limit is not None:
            self.ik_limits.append(self.collision_avoidance_limit)
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    @staticmethod
    def apply_task_profile(ik_config, task_profile):
        profiles = ik_config.get("task_profiles", {})
        if task_profile is None:
            task_profile = ik_config.get("default_task_profile")
        if task_profile is None:
            return None
        if task_profile not in profiles:
            choices = ", ".join(sorted(profiles)) or "none"
            raise ValueError(
                f"unknown task profile {task_profile!r}; available profiles: {choices}"
            )
        for table_name, table_overrides in profiles[task_profile].items():
            if table_name not in {"ik_match_table1", "ik_match_table2"}:
                raise ValueError(
                    f"task profile {task_profile!r} cannot override {table_name!r}"
                )
            table = ik_config[table_name]
            for frame_name, costs in table_overrides.items():
                if frame_name not in table:
                    raise ValueError(
                        f"task profile {task_profile!r} references unknown frame "
                        f"{frame_name!r}"
                    )
                unknown = set(costs) - {"position_cost", "orientation_cost"}
                if unknown:
                    raise ValueError(
                        f"unsupported task-cost overrides for {frame_name!r}: "
                        + ", ".join(sorted(unknown))
                    )
                if "position_cost" in costs:
                    table[frame_name][1] = costs["position_cost"]
                if "orientation_cost" in costs:
                    table[frame_name][2] = costs["orientation_cost"]
        return task_profile

    def make_collision_avoidance_limit(self, config):
        if not config or not config.get("enabled", False):
            return None
        collision_geoms = [
            self.model.geom(geom_id).name
            for geom_id in range(self.model.ngeom)
            if self.model.geom_contype[geom_id]
            and self.model.geom_conaffinity[geom_id]
            and self.model.geom(geom_id).name
        ]
        if not collision_geoms:
            raise ValueError("collision avoidance found no named collision geoms")
        if config.get("all_collision_geoms", False):
            geom_pairs = [(collision_geoms, collision_geoms)]
        else:
            geom_pairs = []
            for pair in config.get("geom_pairs", []):
                if len(pair) != 2:
                    raise ValueError("each collision geom pair must contain two groups")
                resolved_pair = []
                for group in pair:
                    resolved_group = []
                    for selector in group:
                        if selector.startswith("body:"):
                            body_id = self.model.body(selector.removeprefix("body:")).id
                            resolved_group.extend(
                                self.model.geom(geom_id).name
                                for geom_id in range(self.model.ngeom)
                                if self.model.geom_bodyid[geom_id] == body_id
                                and self.model.geom_contype[geom_id]
                                and self.model.geom_conaffinity[geom_id]
                                and self.model.geom(geom_id).name
                            )
                        else:
                            self.model.geom(selector)
                            resolved_group.append(selector)
                    if not resolved_group:
                        raise ValueError(
                            f"collision selector group {group!r} resolved no geoms"
                        )
                    resolved_pair.append(resolved_group)
                geom_pairs.append(tuple(resolved_pair))
            if not geom_pairs:
                raise ValueError(
                    "collision avoidance requires all_collision_geoms or geom_pairs"
                )
        return mink.CollisionAvoidanceLimit(
            self.model,
            geom_pairs,
            gain=float(config.get("gain", 0.85)),
            minimum_distance_from_collisions=float(
                config.get("minimum_distance", 0.005)
            ),
            collision_detection_distance=float(
                config.get("detection_distance", 0.01)
            ),
            bound_relaxation=float(config.get("bound_relaxation", 0.0)),
        )

    def velocity_limited_joint_names(self):
        names = []
        for actuator_id in range(self.model.nu):
            transmission = self.model.actuator_trntype[actuator_id]
            target_id = int(self.model.actuator_trnid[actuator_id, 0])
            if transmission in {
                mj.mjtTrn.mjTRN_JOINT,
                mj.mjtTrn.mjTRN_JOINTINPARENT,
            }:
                joint_name = self.model.joint(target_id).name
            elif transmission == mj.mjtTrn.mjTRN_TENDON and self.tgt_robot == "bello":
                tendon_name = self.model.tendon(target_id).name
                joint_name = BELLO_ANKLE_TENDON_LOGICAL_JOINT.get(tendon_name)
            else:
                joint_name = None
            if joint_name is None:
                actuator_name = self.model.actuator(actuator_id).name
                raise ValueError(
                    f"actuator {actuator_name!r} has no velocity-limit joint mapping"
                )
            names.append(joint_name)
        if len(set(names)) != len(names):
            raise ValueError("velocity-limit actuator joint mappings must be unique")
        return tuple(names)

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets1[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = R.from_quat(
                rot_offset, scalar_first=True
            )
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.tasks2.append(task)
                self.task_errors2[task] = []

        if self.planar_relative_yaw_config is not None:
            config = self.planar_relative_yaw_config
            required_keys = {
                "robot_frame_name",
                "robot_root_name",
                "robot_joint_name",
                "human_frame_landmarks",
                "human_root_landmarks",
                "orientation_cost",
            }
            missing_keys = required_keys - set(config)
            if missing_keys:
                raise ValueError(
                    "planar relative-yaw task is missing keys: "
                    + ", ".join(sorted(missing_keys))
                )
            for key in ("human_frame_landmarks", "human_root_landmarks"):
                landmarks = config[key]
                if len(landmarks) != 2 or any(
                    landmark not in self.human_scale_table for landmark in landmarks
                ):
                    raise ValueError(f"{key} must contain two scaled human landmarks")
            joint = self.model.joint(config["robot_joint_name"])
            if joint.type != mj.mjtJoint.mjJNT_HINGE:
                raise ValueError("planar relative-yaw task requires a hinge joint")
            orientation_cost = float(config["orientation_cost"])
            if not np.isfinite(orientation_cost) or orientation_cost <= 0.0:
                raise ValueError(
                    "planar relative-yaw orientation cost must be positive"
                )
            self.planar_relative_yaw_task = mink.RelativeFrameTask(
                frame_name=config["robot_frame_name"],
                frame_type="body",
                root_name=config["robot_root_name"],
                root_type="body",
                position_cost=0.0,
                orientation_cost=[0.0, 0.0, orientation_cost],
                lm_damping=1,
            )
            self.planar_relative_yaw_reference = mink.Configuration(self.model)
            self.planar_relative_yaw_joint_qpos_address = int(joint.qposadr[0])
            self.tasks2.append(self.planar_relative_yaw_task)
            self.task_errors2[self.planar_relative_yaw_task] = []

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.align_human_data_to_ground(human_data)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
            self.update_planar_relative_yaw_target(human_data)
            
            
    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration,
                self.tasks1,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration,
                    self.tasks1,
                    dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration,
                self.tasks2,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration,
                    self.tasks2,
                    dt,
                    self.solver,
                    self.damping,
                    limits=self.ik_limits,
                )
                self.configuration.integrate_inplace(vel2, dt)
                next_error = self.error2()
                num_iter += 1

        self.enforce_ground_clearance()
        return self.configuration.data.qpos.copy()

    def update_planar_relative_yaw_target(self, human_data):
        if self.planar_relative_yaw_task is None:
            return
        config = self.planar_relative_yaw_config
        frame_left, frame_right = config["human_frame_landmarks"]
        root_left, root_right = config["human_root_landmarks"]
        frame_axis = human_data[frame_right][0] - human_data[frame_left][0]
        root_axis = human_data[root_right][0] - human_data[root_left][0]
        frame_axis = np.asarray(frame_axis[:2], dtype=float)
        root_axis = np.asarray(root_axis[:2], dtype=float)
        frame_norm = np.linalg.norm(frame_axis)
        root_norm = np.linalg.norm(root_axis)
        if frame_norm < 1e-8 or root_norm < 1e-8:
            raise ValueError("planar relative-yaw landmarks must span a direction")
        frame_axis /= frame_norm
        root_axis /= root_norm
        cross = root_axis[0] * frame_axis[1] - root_axis[1] * frame_axis[0]
        target_yaw = np.arctan2(cross, np.dot(root_axis, frame_axis))
        target_yaw *= float(config.get("scale", 1.0))

        joint = self.model.joint(config["robot_joint_name"])
        if joint.limited:
            target_yaw = np.clip(target_yaw, joint.range[0], joint.range[1])
        qpos = self.model.qpos0.copy()
        qpos[self.planar_relative_yaw_joint_qpos_address] = target_yaw
        self.planar_relative_yaw_reference.update(qpos)
        self.planar_relative_yaw_task.set_target(
            self.planar_relative_yaw_reference.get_transform(
                config["robot_frame_name"],
                "body",
                config["robot_root_name"],
                "body",
            )
        )

    def enforce_ground_clearance(self):
        if not self.ground_clearance_geom_ids:
            return
        lowest_height = min(
            self.configuration.data.geom_xpos[geom_id, 2]
            - np.sum(
                np.abs(
                    self.configuration.data.geom_xmat[geom_id].reshape(3, 3)[2]
                )
                * self.model.geom_size[geom_id]
            )
            for geom_id in self.ground_clearance_geom_ids
        )
        ground_height = float(self.ground[2])
        if lowest_height < ground_height - 1e-10:
            qpos = self.configuration.data.qpos.copy()
            qpos[self.root_height_qpos_address] += ground_height - lowest_height
            self.configuration.update(qpos)


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def align_human_data_to_ground(self, human_data):
        world_up = np.array([0.0, 0.0, 1.0])
        for body_name, local_axis in self.ground_alignment_axes.items():
            position, quaternion = human_data[body_name]
            target_rotation = R.from_quat(quaternion, scalar_first=True)
            current_normal = target_rotation.apply(local_axis)
            dot = np.clip(np.dot(current_normal, world_up), -1.0, 1.0)
            if dot < -1.0 + 1e-8:
                perpendicular = np.cross(current_normal, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(perpendicular) < 1e-8:
                    perpendicular = np.cross(
                        current_normal, np.array([0.0, 1.0, 0.0])
                    )
                correction = R.from_rotvec(
                    np.pi * perpendicular / np.linalg.norm(perpendicular)
                )
            else:
                correction_quaternion = np.concatenate(
                    ([1.0 + dot], np.cross(current_normal, world_up))
                )
                correction_quaternion /= np.linalg.norm(correction_quaternion)
                correction = R.from_quat(
                    correction_quaternion, scalar_first=True
                )
            human_data[body_name] = [
                position,
                (correction * target_rotation).as_quat(scalar_first=True),
            ]
        return human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
