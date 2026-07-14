
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

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
        use_velocity_limit: bool | None=None,
        source_fps: float=30.0,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
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
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])
        if use_velocity_limit is None:
            use_velocity_limit = ik_config.get("use_velocity_limit", False)
        if source_fps <= 0.0:
            raise ValueError("source_fps must be positive")
        self.source_fps = source_fps
        self.max_joint_velocity = ik_config.get("max_joint_velocity", 3 * np.pi)
        self.velocity_limited_qpos_addresses = []
        self.previous_output_qpos = None

        self.max_iter = ik_config.get("max_iter", 10)

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
        if use_velocity_limit:
            actuator_joint_names = {
                self.model.joint(self.model.actuator_trnid[actuator_id, 0]).name
                for actuator_id in range(self.model.nu)
            }
            VELOCITY_LIMITS = {
                joint_name: self.max_joint_velocity
                for joint_name in actuator_joint_names
            }
            self.velocity_limited_qpos_addresses = [
                int(self.model.joint(joint_name).qposadr[0])
                for joint_name in actuator_joint_names
            ]
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
        self.add_collision_avoidance_limit(ik_config.get("collision_avoidance"))
            
        self.setup_retarget_configuration()
        self.add_posture_task(ik_config.get("posture_costs"))
        
        self.ground_offset = 0.0

    def add_collision_avoidance_limit(self, config):
        if config is None:
            return

        if not config.get("use_model_contact_matrix", False):
            raise ValueError(
                "collision_avoidance requires use_model_contact_matrix=true"
            )

        self.ik_limits.append(
            mink.CollisionAvoidanceLimit(
                self.model,
                self.model_collision_geom_pairs(),
                gain=config.get("gain", 0.85),
                minimum_distance_from_collisions=config.get(
                    "minimum_distance", 0.0
                ),
                collision_detection_distance=config.get(
                    "detection_distance", 0.05
                ),
                bound_relaxation=config.get("bound_relaxation", 0.0),
            )
        )

    def model_collision_geom_pairs(self):
        """Return self-collision pairs allowed by the compiled MuJoCo model."""
        model = self.model
        excluded_bodies = set(model.exclude_signature.tolist())
        filter_parent = not (
            model.opt.disableflags & mj.mjtDisableBit.mjDSBL_FILTERPARENT
        )
        collidable_geoms = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
        ]
        geom_pairs = []
        for index, geom1 in enumerate(collidable_geoms):
            body1 = int(model.geom_bodyid[geom1])
            if body1 == 0:
                continue
            weld1 = int(model.body_weldid[body1])
            allowed_geom2 = []
            for geom2 in collidable_geoms[index + 1 :]:
                body2 = int(model.geom_bodyid[geom2])
                if body2 == 0:
                    continue
                weld2 = int(model.body_weldid[body2])
                if weld1 == weld2:
                    continue
                if filter_parent and (
                    int(model.body_parentid[weld1]) == weld2
                    or int(model.body_parentid[weld2]) == weld1
                ):
                    continue
                if not (
                    model.geom_contype[geom1] & model.geom_conaffinity[geom2]
                    or model.geom_contype[geom2] & model.geom_conaffinity[geom1]
                ):
                    continue
                body_signature = (min(body1, body2) << 16) + max(body1, body2)
                if body_signature in excluded_bodies:
                    continue
                allowed_geom2.append(geom2)
            if allowed_geom2:
                geom_pairs.append(([geom1], allowed_geom2))
        return geom_pairs

    def add_posture_task(self, posture_costs):
        if posture_costs is None:
            return
        costs = np.zeros(self.model.nv)
        for joint_name, cost in posture_costs.items():
            joint = self.model.joint(joint_name)
            costs[int(joint.dofadr[0])] = cost
        posture_task = mink.PostureTask(self.model, cost=costs)
        posture_task.set_target(self.model.qpos0)
        self.tasks1.append(posture_task)
        self.tasks2.append(posture_task)

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
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
                # Solve the IK problem with the second task
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
                
            
        qpos = self.limit_output_velocity(self.configuration.data.qpos.copy())
        self.configuration.update(qpos)
        return qpos

    def limit_output_velocity(self, qpos):
        if self.velocity_limited_qpos_addresses and self.previous_output_qpos is not None:
            addresses = self.velocity_limited_qpos_addresses
            max_delta = self.max_joint_velocity / self.source_fps
            qpos[addresses] = np.clip(
                qpos[addresses],
                self.previous_output_qpos[addresses] - max_delta,
                self.previous_output_qpos[addresses] + max_delta,
            )
        self.previous_output_qpos = qpos.copy()
        return qpos


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
            if body_name not in rot_offsets:
                continue
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

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
