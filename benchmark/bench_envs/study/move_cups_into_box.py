# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
import numpy as np
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox, get_collison_with_objs
from bench_envs.utils.scene_gen_utils import print_c, place_actor
from transforms3d.euler import euler2quat

class move_cups_into_box(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {t[-1].get_name() for t in self.target_objects}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
                #place obstacles inside and next to the box
        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            self.obstacle_density = max(0, self.obstacle_density-1)  # Ensure non-negative density
            des_bb = get_actor_boundingbox(self.box.actor)
            box_obs = "001_bottle"
            gap = 0.05
            y_gap = 0.08
            bid = np.random.choice(task_objs["objects"]["study"]["obstacles"]["tall"][box_obs])
            place_pose =  [[des_bb[1][0]+gap if self.scene_id == 0 else des_bb[0][0]-gap, 
                           des_bb[1][1]-np.random.uniform(low=y_gap, 
                                    high=des_bb[1][1]-des_bb[0][1] - y_gap),
                           des_bb[0][-1]],(90,0,0)]
            bid = np.random.choice(task_objs["objects"]["study"]["obstacles"]["tall"][box_obs])
            box_obs_tar, obs_tar_id, _= place_actor(box_obs, self, 
                           task_objs = task_objs, obj_id = bid,
                          obj_pose=place_pose, mass = 0.5, is_static=False)
            self.collision_list.append({
                "actor":box_obs_tar,
                "collision_path": self.col_temp.format(object=box_obs,
                                                        object_id=obs_tar_id)
            })
        xlim, ylim, self.side_to_place = get_position_limits(self.table, boundary_thr=0.20, 
                                                             side="right" if self.scene_id else "left")
      
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
    
        self.target_objects = []
        self.des_obj_poses = []
        des_pos = self.box.get_pose().p
        box_bb = get_actor_boundingbox(self.box.actor)
        place_gap = 0.10
        self.target_name = "021_cup"
        for i, tn in enumerate(["021_cup","021_cup"]):
            target_obj, target_id, target_pose = \
            place_actor(tn, self, col_thr=0.10, xlim=xlim, ylim=ylim, 
                        qpos=(90,0,90), object_bounds=object_bounds, task_objs=task_objs,
                        mass = 0.1, rotation=False, obj_id=None)

            self.target_objects.append((tn, target_id, target_obj))
            self.des_obj_poses.append([des_pos[0], des_pos[1] - place_gap + (i*place_gap), box_bb[1][-1]]+[1,0,0,0])
            self.add_prohibit_area(target_obj, padding=0.08, area="table")

        print_c(f"Placement destination pose {des_pos}", "RED")

      
    def play_once(self, z = 0.1, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place )

        for t, d in zip(self.target_objects,self.des_obj_poses):
            # Grasp the mouse with the selected arm
            self.move(self.grasp_actor(t[-1], arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
            x = z if self.side_to_place == "right" else -z
            # Lift the mouse upward by 0.1 meters in z-direction
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=x, z=z,
                                                constraint_pose =[0,0,0.8, 0, 0, 0]))

            self.attach_object(t[-1], f"{os.environ['ASSETS_ROOT']}/objects/{t[0]}/collision/base{t[1]}.glb", str(arm_tag))

            # Place the mouse at the target location with alignment constraint
            self.move(
                self.place_actor(
                    t[-1],
                    arm_tag=arm_tag,
                    target_pose= d,
                    constrain="auto",
                    pre_dis=pre_dis,
                    dis=dis,
                ))
            self.detach_object(arm_tag)
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{2}",
            "{B}": f"red",
            "{a}": str(arm_tag)
        }
        return self.info

    def check_success(self):
        box_bb = get_actor_boundingbox(self.box.actor)
        obj_poses = np.stack([op[-1].get_pose().p for op in self.target_objects], axis=0)  # full XYZ
        xy_in = np.all((box_bb[0][:2] <= obj_poses[:, :2]) & (obj_poses[:, :2] <= box_bb[1][:2]), axis=1)
        below_rim = obj_poses[:, 2] < box_bb[1][2]     # each cup below the box rim, not balanced on the wall
        return (bool(np.all(xy_in & below_rim))
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
