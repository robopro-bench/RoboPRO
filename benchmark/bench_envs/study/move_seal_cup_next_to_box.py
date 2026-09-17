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
from bench_envs.utils.scene_gen_utils import print_c, place_actor, point_to_box_distance
from transforms3d.euler import euler2quat

class move_seal_cup_next_to_box(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.target_obj_2.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                      boundary_thr=0.10, side= "left" if self.scene_id == 0 else "right")
        
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
        # Place a cup as target
 
        self.target_name = "100_seal"
        self.move_thr = 0.08

        xlim_s = [xlim[0] + self.move_thr + 0.15, xlim[1]] if self.scene_id == 0 \
        else [xlim[0], xlim[1] - self.move_thr - 0.15]
        ylim_cons = [ylim[0], ylim[1]-0.05]

        # print(f"Seal initial pose range: x={xlim_s}, y={ylim_cons}")

        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.15, xlim=xlim_s, ylim=ylim_cons, 
                    qpos=(90,0,0), object_bounds=object_bounds, task_objs=task_objs,
                     mass = 0.2, rotation=False)
        object_bounds.append(get_actor_boundingbox(self.target_obj.actor))
        bb_box = get_actor_boundingbox(self.box.actor)        
        self.des_obj_pose = [bb_box[1][0] + self.move_thr if self.side_to_place == "left" else bb_box[0][0] - self.move_thr,
             np.random.uniform(low=bb_box[0][1]+0.05, high=bb_box[1][1]-0.10),
             bb_box[1][-1]-0.02] + [1,0,0,0]

        # Place a cup as target
        self.target_name_2 = "021_cup"

        xlim_c = [xlim[0] + 0.1, xlim[1]] if self.scene_id == 0 \
        else [xlim[0], xlim[1] - self.move_thr]
        # print(f"Cup initial pose range: x={xlim_c}, y={ylim_cons}")
        self.target_obj_2, self.target_id_2, self.target_pose_2 = \
        place_actor(self.target_name_2, self, col_thr=0.15, xlim=xlim_c, ylim=ylim_cons, 
                    qpos=(90,0,90), object_bounds=object_bounds, task_objs=task_objs,
                     mass = 0.1, rotation=False)


        self.add_prohibit_area(sapien.Pose(p = self.des_obj_pose[:3], 
                                           q = [1,0,0,0]), 
                                           padding=0.12, area="table")

        p = self.box.get_pose().p.tolist() 
        p[1] -= 0.1
        p[2] = bb_box[1][-1] -0.05 

        self.des_obj_pose_2 = p + [1, 0, 0, 0]

        self.add_prohibit_area(self.target_obj, padding=0.1, area="table")
        self.add_prohibit_area(self.target_obj_2, padding=0.1, area="table")
       
        if np.random.rand() > self.clean_background_rate:
            box_obs = "001_bottle"
            gap = 0.1
            place_pose =  [[np.random.choice([bb_box[0][0]+gap,
                                              bb_box[1][0]-gap]), 
                        bb_box[1][1]-gap,
                        bb_box[0][-1]],(90,0,0)]
            
            box_obs_tar, obs_tar_id, _= place_actor(box_obs, self, 
                           task_objs = task_objs, obj_id = 0,
                          obj_pose=place_pose, mass = 0.5, is_static=False)
            self.collision_list.append({
                "actor":box_obs_tar,
                "collision_path": self.col_temp.format(object=box_obs,
                                                        object_id=obs_tar_id)
            })

        print_c(f"Placement seal pose {self.des_obj_pose}", "RED")
        print_c(f"Placement cup pose {self.des_obj_pose_2}", "RED")      

    def move_object(self, arm_tag, target_obj, target_id, target_name, 
                    des_obj_pose, z = 0.1, pre_dis= 0.07, 
                    dis=0.005, pre_grasp_dist=0.1):

        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        # Lift the mouse upward by 0.1 meters in z-direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=-z if self.side_to_place == "left" else z, z=z))

        self.attach_object(target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{target_name}/collision/base{target_id}.glb", str(arm_tag))
        # Place the mouse at the target location with alignment constraint
        self.move(
            self.place_actor(
                target_obj,
                arm_tag=arm_tag,
                target_pose= des_obj_pose,
                constrain= "free",
                pre_dis=pre_dis,
                dis=dis,
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))
        self.detach_object(arm_tag)

    def play_once(self):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) 
        self.move_object(arm_tag, self.target_obj, self.target_id, 
                         self.target_name, self.des_obj_pose,
                        z = 0.1, pre_dis= 0.07, 
                        dis=0.005, pre_grasp_dist=0.08)
        self.move_object(arm_tag, self.target_obj_2, self.target_id_2, 
                         self.target_name_2, self.des_obj_pose_2,
                        z = 0.1, pre_dis= 0.07, 
                        dis=0.005, pre_grasp_dist=0.08)
        # Record information about the objects and arm used in the task
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info


    def check_success(self):
        dist_thr = 0.15

        box_bb = get_actor_boundingbox(self.box.actor)

        seal_pose = self.target_obj.get_pose().p
        cup_pose = self.target_obj_2.get_pose().p

        cup_in_box = (np.all((box_bb[0][:2] <= cup_pose[:2])  &  (cup_pose[:2] <= box_bb[1][:2]))
                      and cup_pose[2] < box_bb[1][2])   # below the box rim, not balanced on the wall
        seal_in_box = np.all((box_bb[0][:2] <= seal_pose[:2])  &  (seal_pose[:2] <= box_bb[1][:2]))
        dist_to_box = point_to_box_distance(seal_pose, box_bb[0], box_bb[1])
        return (cup_in_box and not seal_in_box and dist_to_box <= dist_thr 
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())

