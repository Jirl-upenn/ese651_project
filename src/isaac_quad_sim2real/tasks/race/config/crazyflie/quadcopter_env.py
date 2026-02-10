# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from matplotlib import pyplot as plt
from collections import deque
import math

from pxr import Gf, UsdGeom, Sdf, UsdPhysics, PhysxSchema
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat

from typing import List
from dataclasses import dataclass, field
from datetime import datetime
import csv

from scipy.spatial.transform import Rotation as R

# Import strategy class
from .quadcopter_strategies import DefaultQuadcopterStrategy

##
# Drone config
##
# from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    collision_group=0,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"usd/cf2x.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

D2R = np.pi / 180.0
R2D = 180.0 / np.pi
PLOT_UPDATE_FREQ = 5


GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.
        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@dataclass
class GateModelCfg:
    usd_path: str = "./usd/gate.usda"
    prim_name: str = "gate"
    gate_side: float = 1.0
    scale = [1.0, gate_side, gate_side]

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    use_wall = False
    track_name = 'powerloop'

    # env
    episode_length_s = 30.0             # episode_length = episode_length_s / dt / decimation
    action_space = 4
    observation_space = 1 # inconsequential, just needs to exist for Gymnasium compatibility
    state_space = 0
    debug_vis = True

    sim_rate_hz = 500
    policy_rate_hz = 50
    pid_loop_rate_hz = 500
    decimation = sim_rate_hz // policy_rate_hz
    pid_loop_decimation = sim_rate_hz // pid_loop_rate_hz

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)
    gate_model: GateModelCfg = field(default_factory=GateModelCfg)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        force_threshold=0.0,
        # filter_prim_paths_expr=["/World/envs/env_.*/Wall"],
    )

    beta = 1.0         # 1.0 for no smoothing, 0.0 for no update

    # Reset variables
    min_altitude = 0.1
    max_altitude = 3.0
    max_time_on_ground = 1.5

    # motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    # PID parameters
    kp_omega_rp = 250.0
    ki_omega_rp = 500.0
    kd_omega_rp = 2.5
    i_limit_rp = 33.3

    kp_omega_y = 120.0
    ki_omega_y = 16.70
    kd_omega_y = 0.0
    i_limit_y = 166.7

    body_rate_scale_xy = 100.0 * D2R
    body_rate_scale_z = 200.0 * D2R

    # Parameters from train.py or play.py
    is_train = None

    k_aero_xy = 9.1785e-7
    k_aero_z = 10.311e-7

    max_tilt_thresh = 150 * D2R

    max_n_laps = 3

    rewards = {}

    # Strategy class for custom rewards, observations, and resets
    strategy_class: type[DefaultQuadcopterStrategy] = DefaultQuadcopterStrategy

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        self._all_target_models_paths: List[List[str]] = []
        self._models_paths_initialized: bool = False
        self.target_models_prim_base_name: str | None = None

        super().__init__(cfg, render_mode, **kwargs)

        self.iteration = 0

        if len(cfg.rewards) > 0:
            self.rew = cfg.rewards
        elif self.cfg.is_train:
            raise ValueError("rewards not provided")

        # Initialize tensors
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_yaw = torch.zeros(self.num_envs, device=self.device)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_omega_meas = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._last_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._yaw_n_laps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._idx_wp = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._n_gates_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._crashed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # Motor dynamics
        self.cfg.thrust_to_weight = 3.15
        r = self.cfg.arm_length * np.sqrt(2.0) / 2.0
        self._rotor_positions = torch.tensor(
            [
                [ r,  r, 0],
                [ r, -r, 0],
                [-r, -r, 0],
                [-r,  r, 0]
            ],
            dtype=torch.float32,
            device=self.device
        )
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device)
        self.k = self.cfg.k_m / self.cfg.k_eta

        self.f_to_TM = torch.cat(
            [
                torch.tensor([[1, 1, 1, 1]], device=self.device),
                torch.cat(
                    [
                        torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
                    ],
                    dim=1,
                ).to(self.device),
                self.k * self._rotor_directions.view(1, -1),
            ],
            dim=0
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)

        # Initialize parameter tensors and values (must be done before strategy initialization)
        self._K_aero = torch.zeros(self.num_envs, 3, device=self.device)
        self._kp_omega = torch.zeros(self.num_envs, 3, device=self.device)
        self._ki_omega = torch.zeros(self.num_envs, 3, device=self.device)
        self._kd_omega = torch.zeros(self.num_envs, 3, device=self.device)
        self._tau_m = torch.zeros(self.num_envs, 4, device=self.device)
        self._omega_err_integral = torch.zeros(self.num_envs, 3, device=self.device)
        self._thrust_to_weight = torch.zeros(self.num_envs, device=self.device)

        # Store fixed parameter values
        self._twr_value = self.cfg.thrust_to_weight
        self._k_aero_xy_value = self.cfg.k_aero_xy
        self._k_aero_z_value = self.cfg.k_aero_z
        self._kp_omega_rp_value = self.cfg.kp_omega_rp
        self._ki_omega_rp_value = self.cfg.ki_omega_rp
        self._kd_omega_rp_value = self.cfg.kd_omega_rp
        self._kp_omega_y_value = self.cfg.kp_omega_y
        self._ki_omega_y_value = self.cfg.ki_omega_y
        self._kd_omega_y_value = self.cfg.kd_omega_y
        self._tau_m_value = self.cfg.tau_m

        # Initialize the strategy for rewards, observations, and resets
        # Strategy __init__ may set fixed parameter values using the _value attributes above
        self.strategy = self.cfg.strategy_class(self)

        # Initialize other state variables
        self._pose_drone_wrt_gate = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)
        self._initial_wp = 0
        self._n_run = 0

        self.set_debug_vis(self.cfg.debug_vis)

    def update_iteration(self, iter):
        self.iteration = iter

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = GOAL_MARKER_CFG.copy()
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._gate_model_cfg_data = getattr(self.cfg, 'gate_model', {})
        model_usd_file_path = self._gate_model_cfg_data.usd_path

        self._target_models_prim_base_name = self._gate_model_cfg_data.prim_name

        model_scale = Gf.Vec3f(*self._gate_model_cfg_data.scale)

        stage = get_current_stage()
        env0_root_path_str = "/World/envs/env_0"

        d = self._gate_model_cfg_data.gate_side / 2
        self._local_square = torch.tensor([
            [0,  d,  d],
            [0, -d,  d],
            [0, -d, -d],
            [0,  d, -d]
        ], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1, 1)

        #########################

        tracks = {
            'complex': [
                [ 1.5,  3.5, 0.75, 0.0, 0.0, -0.7854],
                [-1.5,  3.5, 0.75, 0.0, 0.0,  0.7854],
                [-2.0, -3.5, 2.00, 0.0, 0.0,  1.5708],
                [-2.0, -3.5, 0.75, 0.0, 0.0, -1.5708],
                [ 1.0, -1.0, 2.00, 0.0, 0.0,  3.1415],
                [ 1.0, -3.5, 0.75, 0.0, 0.0,  0.0000],
            ],
            'powerloop': [
                [2.0, 3.5, 0.75, 0.0, 0.0, -1.5708],
                [-1.5, 3.5, 2.00, 0.0, 0.0, 0.7854],
                [-0.625, 0.0, 0.75, 0.0, 0.0, 1.5708],
                [0.625, 0.0, 0.75, 0.0, 0.0, 1.5708],
                [-1.5, -3.5, 2.00, 0.0, 0.0, 2.356],
                [2.0, -3.5, 0.75, 0.0, 0.0, -1.5708],
                [0.625, 0.0, 0.75, 0.0, 0.0, -1.5708],
            ],
            'lemniscate': [
                [ 1.5, 3.50, 0.75, 0.0, 0.0, -1.57],
                [ 0.0, 5.25, 1.50, 0.0, 0.0,  0.00],
                [-2.0, 7.00, 0.75, 0.0, 0.0, -1.57],
                [ 1.5, 7.00, 0.75, 0.0, 0.0,  1.57],
                [ 0.0, 5.25, 1.50, 0.0, 0.0,  0.00],
                [-2.0, 3.50, 0.75, 0.0, 0.0,  1.57],
            ]
        }

        self._waypoints = torch.tensor(tracks[self.cfg.track_name], device=self.device)

        self._normal_vectors = torch.zeros(self._waypoints.shape[0], 3, device=self.device)
        self._waypoints_quat = torch.zeros(self._waypoints.shape[0], 4, device=self.device)

        if not stage.GetPrimAtPath(env0_root_path_str):
            UsdGeom.Xform.Define(stage, Sdf.Path(env0_root_path_str))

        for i, waypoint_data in enumerate(self._waypoints):
            position_tensor = waypoint_data[0:3]
            euler_angles_tensor = waypoint_data[3:6]

            euler_np = euler_angles_tensor.cpu().numpy()
            rot_from_euler = R.from_euler('xyz', euler_np)
            quat_xyzw = rot_from_euler.as_quat()  # shape: (4,) as [x, y, z, w]
            quat_wxyz = np.roll(quat_xyzw, shift=1)  # now [w, x, y, z]
            self._waypoints_quat[i, :] = torch.tensor(quat_wxyz, device=self.device, dtype=torch.float32)
            rotmat_np_gate = rot_from_euler.as_matrix()
            gate_normal_np = rotmat_np_gate[:, 0] 
            self._normal_vectors[i, :] = torch.tensor(gate_normal_np, device=self.device, dtype=torch.float32)
            current_gate_normal_world = Gf.Vec3d(float(gate_normal_np[0]), float(gate_normal_np[1]), float(gate_normal_np[2])).GetNormalized()

            current_gate_pose_position = Gf.Vec3d(
                float(position_tensor[0]), float(position_tensor[1]), float(position_tensor[2])
            )
            quat_numpy_array_gate = euler_angles_to_quat(euler_angles_tensor.cpu().numpy())
            current_gate_pose_orientation_gd = Gf.Quatd(
                float(quat_numpy_array_gate[0]), float(quat_numpy_array_gate[1]),
                float(quat_numpy_array_gate[2]), float(quat_numpy_array_gate[3])
            )

            model_pose_xform_name = f"{self._target_models_prim_base_name}_{i}"
            model_pose_xform_path = f"{env0_root_path_str}/{model_pose_xform_name}"
            scaled_ref_xform_name = "scaled_model_ref"
            scaled_ref_xform_path = f"{model_pose_xform_path}/{scaled_ref_xform_name}"

            # 1. Create external Xform for the model pose
            usd_geom_pose_xform_obj = UsdGeom.Xform.Define(stage, Sdf.Path(model_pose_xform_path))
            model_pose_xform_prim = usd_geom_pose_xform_obj.GetPrim()

            if not model_pose_xform_prim or not model_pose_xform_prim.IsValid():
                continue

            xformable_pose_gate = UsdGeom.Xformable(model_pose_xform_prim)
            xformable_pose_gate.ClearXformOpOrder()
            op_orient_pose_gate = xformable_pose_gate.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            op_orient_pose_gate.Set(current_gate_pose_orientation_gd)
            op_translate_pose_gate = xformable_pose_gate.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            op_translate_pose_gate.Set(current_gate_pose_position)
            xformable_pose_gate.SetXformOpOrder([op_translate_pose_gate, op_orient_pose_gate])

            # 2. Create Xform for the scaled reference model
            usd_geom_scaled_ref_xform_obj = UsdGeom.Xform.Define(stage, Sdf.Path(scaled_ref_xform_path))
            model_scaled_ref_xform_prim = usd_geom_scaled_ref_xform_obj.GetPrim()
            if not model_scaled_ref_xform_prim or not model_scaled_ref_xform_prim.IsValid():
                continue

            xformable_scaled_ref_gate = UsdGeom.Xformable(model_scaled_ref_xform_prim)
            xformable_scaled_ref_gate.ClearXformOpOrder()
            op_scale_model_gate = xformable_scaled_ref_gate.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            op_scale_model_gate.Set(model_scale)
            xformable_scaled_ref_gate.SetXformOpOrder([op_scale_model_gate])

            # 3. Create gates
            references_api_gate = model_scaled_ref_xform_prim.GetReferences()
            references_api_gate.AddReference(assetPath=model_usd_file_path)

            # 4. Apply collisions to gates
            for child_prim in model_scaled_ref_xform_prim.GetChildren():
                for mesh_prim in child_prim.GetChildren():
                    if mesh_prim.GetTypeName() == "Mesh":
                        # Apply rigid body to the parent xform
                        rb_api = UsdPhysics.RigidBodyAPI.Apply(child_prim)
                        rb_api.CreateKinematicEnabledAttr().Set(True)

                        # apply collision API to the mesh
                        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
                        collision_api.CreateCollisionEnabledAttr().Set(True)

                        # apply mesh collision API with convex decomposition
                        # this creates multiple convex shapes that preserve the gate opening
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
                        mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")

            arrow_length = 0.5
            arrow_body_radius = 0.01
            arrow_head_radius = 0.03
            arrow_head_height_factor = 0.25
            arrow_color_gf = Gf.Vec3f(0.0, 1.0, 0.0)

            arrow_xform_name = f"{model_pose_xform_name}_normal_arrow"
            arrow_xform_path = f"{env0_root_path_str}/{arrow_xform_name}"

            arrow_parent_xform_geom = UsdGeom.Xform.Define(stage, Sdf.Path(arrow_xform_path))
            arrow_parent_xform_prim = arrow_parent_xform_geom.GetPrim()

            if arrow_parent_xform_prim and arrow_parent_xform_prim.IsValid():
                default_arrow_up_axis = Gf.Vec3d(0.0, 1.0, 0.0)
                inverted_gate_normal_world = -current_gate_normal_world 

                arrow_rotation = Gf.Rotation(default_arrow_up_axis, inverted_gate_normal_world)
                arrow_orientation_quat = arrow_rotation.GetQuat()

                xformable_arrow_parent = UsdGeom.Xformable(arrow_parent_xform_prim)
                xformable_arrow_parent.ClearXformOpOrder()

                op_orient_arrow = xformable_arrow_parent.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                op_orient_arrow.Set(arrow_orientation_quat)

                op_translate_arrow = xformable_arrow_parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                op_translate_arrow.Set(current_gate_pose_position)

                xformable_arrow_parent.SetXformOpOrder([op_translate_arrow, op_orient_arrow])

                body_height = arrow_length * (1.0 - arrow_head_height_factor)
                head_height = arrow_length * arrow_head_height_factor

                arrow_body_path = f"{arrow_xform_path}/body"
                body_geom_usd = UsdGeom.Cylinder.Define(stage, Sdf.Path(arrow_body_path))
                body_geom_usd.GetAxisAttr().Set(UsdGeom.Tokens.y)
                body_geom_usd.GetRadiusAttr().Set(arrow_body_radius)
                body_geom_usd.GetHeightAttr().Set(body_height)
                UsdGeom.XformCommonAPI(body_geom_usd.GetPrim()).SetTranslate(Gf.Vec3d(0.0, body_height / 2.0, 0.0))

                arrow_head_path = f"{arrow_xform_path}/head"
                head_geom_usd = UsdGeom.Cone.Define(stage, Sdf.Path(arrow_head_path))
                head_geom_usd.GetAxisAttr().Set(UsdGeom.Tokens.y)
                head_geom_usd.GetRadiusAttr().Set(arrow_head_radius)
                head_geom_usd.GetHeightAttr().Set(head_height)
                UsdGeom.XformCommonAPI(head_geom_usd.GetPrim()).SetTranslate(Gf.Vec3d(0.0, body_height + head_height / 2.0, 0.0))

                body_primvars_api = UsdGeom.PrimvarsAPI(body_geom_usd.GetPrim())
                body_primvars_api.CreatePrimvar("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set([arrow_color_gf])

                head_primvars_api = UsdGeom.PrimvarsAPI(head_geom_usd.GetPrim())
                head_primvars_api.CreatePrimvar("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set([arrow_color_gf])

    def _compute_motor_speeds(self, wrench_des):
        f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        motor_speed_squared = f_des / self.cfg.k_eta
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

        return motor_speeds_des

    def _get_moment_from_ctbr(self, actions):
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, 0] = self.cfg.body_rate_scale_xy * actions[:, 1]  # roll_rate
        omega_des[:, 1] = self.cfg.body_rate_scale_xy * actions[:, 2]  # pitch_rate
        omega_des[:, 2] = self.cfg.body_rate_scale_z  * actions[:, 3]  # yaw_rate

        omega_meas = self._robot.data.root_ang_vel_b

        omega_err = omega_des - omega_meas

        self._omega_err_integral += omega_err / self.cfg.pid_loop_rate_hz
        if self.cfg.i_limit_rp > 0 or self.cfg.i_limit_y > 0:
            limits = torch.tensor(
                [self.cfg.i_limit_rp, self.cfg.i_limit_rp, self.cfg.i_limit_y],
                device=self._omega_err_integral.device
            )
            self._omega_err_integral = torch.clamp(
                self._omega_err_integral,
                min=-limits,
                max=limits
            )

        omega_int = self._omega_err_integral

        self._previous_omega_meas = torch.where(
            torch.abs(self._previous_omega_meas) < 0.0001,
            omega_meas,
            self._previous_omega_meas
        )
        omega_meas_dot = (omega_meas - self._previous_omega_meas) * self.cfg.pid_loop_rate_hz
        self._previous_omega_meas = omega_meas.clone()

        omega_dot = (
            self._kp_omega * omega_err +
            self._ki_omega * omega_int -
            self._kd_omega * omega_meas_dot
        )

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    ##########################################################
    ### Functions called in direct_rl_env.py in this order ###
    ##########################################################

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)    # actions come directly from the NN
        self._actions = self.cfg.beta * self._actions + (1 - self.cfg.beta) * self._previous_actions

        # Store current actions for next timestep (for action smoothing and observations)
        self._previous_actions = self._actions.clone()

        self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * self._robot_weight * self._thrust_to_weight
        self.pid_loop_counter = 0

    def _apply_action(self):
        if self.pid_loop_counter % self.cfg.pid_loop_decimation == 0:
            self._wrench_des[:, 1:] = self._get_moment_from_ctbr(self._actions)
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        self.pid_loop_counter += 1

        motor_accel = (self._motor_speeds_des - self._motor_speeds) / self._tau_m
        self._motor_speeds += motor_accel * self.physics_dt

        self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
        motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
        wrench = torch.matmul(motor_forces, self.f_to_TM.t())

        # Compute drag
        lin_vel_b = self._robot.data.root_com_lin_vel_b
        theta_dot = torch.sum(self._motor_speeds, dim=1, keepdim=True)
        drag = -theta_dot * self._K_aero.unsqueeze(0) * lin_vel_b

        self._thrust[:, 0, :] = drag
        self._thrust[:, 0, 2] += wrench[:, 0]
        self._moment[:, 0, :] = wrench[:, 1:]

        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pose = self._robot.data.root_link_state_w[:, :3]
        self._pose_drone_wrt_gate, _ = subtract_frame_transforms(self._waypoints[self._idx_wp, :3],
                                                                 self._waypoints_quat[self._idx_wp, :],
                                                                 drone_pose)

        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        cond_h_min_time = torch.logical_and(
            self._robot.data.root_link_pos_w[:, 2] < self.cfg.min_altitude,
            episode_time > self.cfg.max_time_on_ground
        )

        cond_max_h = self._robot.data.root_link_pos_w[:, 2] > self.cfg.max_altitude

        # self._crashed is computed in get_rewards() in quadcopter_strategies.py.
        cond_crashed = self._crashed > 100

        #TODO ----- START ----- [OPTIONAL]
        # Consider adding additional _get_dones() conditions to influence training. Note that the additional conditions
        # will not be used during runtime for the official class race.
        #TODO ----- END ----- [OPTIONAL]

        died = (
            cond_max_h
          | cond_h_min_time
          | cond_crashed
        )

        # timeout conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.is_train:
            time_out = time_out | ((self._n_gates_passed - 1) // (self._waypoints.shape[0]) >= self.cfg.max_n_laps)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards using the strategy pattern."""
        return self.strategy.get_rewards()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments using the strategy pattern."""
        # Delegate reset logic to strategy
        self.strategy.reset_idx(env_ids)

        # Call parent class reset (required for environment state)
        super()._reset_idx(env_ids)

    def _get_observations(self) -> dict:
        """Get observations using the strategy pattern."""
        return self.strategy.get_observations()
