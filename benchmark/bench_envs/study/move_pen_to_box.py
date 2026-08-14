# from envs._base_task import Base_Task
import sapien
import math
import glob
import yaml
import os
import numpy as np

from bench_envs.study._study_base_task import Study_base_task
from envs.utils import *
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
from bench_envs.utils.scene_gen_utils import get_position_limits, get_actor_boundingbox, get_collison_with_objs
from bench_envs.utils.scene_gen_utils import print_c, place_actor
from transforms3d.euler import euler2quat

class move_pen_to_box(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            des_bb = get_actor_boundingbox(self.box.actor)
            self.obstacle_density = max(0, self.obstacle_density-1) 
            box_obs = "090_trophy"
            gap = 0.05
            y_gap = 0.08
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
            object_bounds.append(get_actor_boundingbox(box_obs_tar.actor))

        xlim, ylim, self.arm_side= get_position_limits(self.table,
                                      boundary_thr=0.15, side="left" if self.scene_id == 0 else "right")
      
        self.des_obj_name = "059_pencup"
        
        self.des_obj, self.des_obj_id, self.des_obj_pose = \
        place_actor(self.des_obj_name, self, col_thr =0.15,
                     xlim=xlim, ylim=ylim, qpos=(90,0,90),
                     object_bounds = object_bounds, task_objs = task_objs,
                     obj_id = 5, mass = 15, rotation=False)
       
    
        des_bb = get_actor_boundingbox(self.des_obj.actor)
        des_bb = get_actor_boundingbox(self.des_obj.actor)
        place_pose =  [[*self.des_obj.get_pose().p[:2],
                         des_bb[0][-1] + 0.03],(90,0,90)]

        self.target_name = "058_markpen"
        self.target_obj, self.target_id, self.target_pose = \
            place_actor(self.target_name, self, task_objs = task_objs,
                    obj_pose=place_pose, mass = 0.1, obj_id=2,
                    qpos=(180,0,0))
       
        self.init_tar_pose = self.target_obj.get_pose()
        self.lift_height = 0.15
        xy_thr = 0.15
        self.ep_lift = xy_thr if self.arm_side == "right" else -xy_thr


        des_bb = get_actor_boundingbox(self.box.actor)

        p = self.box.get_pose().p.tolist() 
        p[-1] = des_bb[1][-1]
        p[0] += 0.05 if self.arm_side == "left" else -0.05
        self.des_obj_pose = p + [1,0,0,0] #self.target_obj.get_pose().q.tolist()
        print_c(f"Placement destination pose {self.des_obj_pose}", "RED")

        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.02, area="table")
      
    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.arm_side) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=self.ep_lift, z=self.lift_height))

        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.des_obj_pose,
                constrain="auto",
                pre_dis=pre_dis,
                dis=dis,
            ))

        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        box_bb = get_actor_boundingbox(self.box.actor)
        p = self.target_obj.get_pose().p
        return (np.all((box_bb[0][:2] <= p[:2]) & (p[:2] <= box_bb[1][:2]))
                # below the box rim: rejects an object balanced on the wall/rim (XY-only used to pass it)
                and p[2] < box_bb[1][2]
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())