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
from bench_envs.utils.scene_gen_utils import print_c, place_actor, get_random_place_pose
from transforms3d.euler import euler2quat


class move_seal_onto_table(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        print_c(self.seed, "YELLOW")
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                         boundary_thr=[0.15, 0.25], 
                                         side="left" if self.scene_id == 0 else "right")
      
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
   
        des_bb = get_actor_boundingbox(self.box.actor)
    
        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            self.obstacle_density = max(0, self.obstacle_density-1) 
            box_obs = "001_bottle"
            gap = 0.05
            y_gap = 0.08
            place_pose =  [[des_bb[1][0]+ gap if self.scene_id == 0 else des_bb[0][0]-gap, 
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
    
        # Object 1
        self.target_name ="100_seal"

        
        box_pose = self.box.get_pose().p
        place_pose =  [[box_pose[0], box_pose[1]-0.05, des_bb[0][-1] + 0.03],(90,0,180)]
       
        self.target_obj, self.target_id, self.target_obj_pose = \
            place_actor(self.target_name, self, task_objs = task_objs,
                    obj_pose=place_pose, mass = 0.1)

        
        
        self.target_place_pose = get_random_place_pose(xlim = [xlim[0] + 0.1, xlim[1] - 0.1], ylim=ylim,
                                             col_thr=0.1, object_bounds=object_bounds)
        
        # Get the placement pose
        self.target_des_pose = self.target_place_pose.p.tolist() + [1,0,0,0]


        print_c(f"Placement destination pose {self.target_des_pose}", "RED")

        self.add_prohibit_area( self.target_place_pose , padding=0.1, area="table")
    
    def play_once(self, z = 0.15, pre_dis= 0.05, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) 
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.move(self.move_by_displacement(arm_tag=arm_tag,x=z if self.side_to_place == "left" else -z, z=z,
                                            constraint_pose=None))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= self.target_des_pose,
                constrain= "free",
                pre_dis=pre_dis,
                dis=dis
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag,  z=z))
        self.info["info"] = {
            "{A}": f"{self.target_name}/base{self.target_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        box_bb = get_actor_boundingbox(self.box.actor)
        seal_pose = self.target_obj.get_pose().p
        eps = 0.03

        seal_in_box = np.all((box_bb[0][:2] <= seal_pose[:2])  &  (seal_pose[:2] <= box_bb[1][:2]))

        table_bb = get_actor_boundingbox(self.table)
        seal_on_table = np.all((table_bb[0][:2] <= seal_pose[:2])  &  (seal_pose[:2] <= table_bb[1][:2]))
        seal_on_table &= (seal_pose[-1] - table_bb[1][-1]) < eps  


        return (not seal_in_box and seal_on_table 
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
