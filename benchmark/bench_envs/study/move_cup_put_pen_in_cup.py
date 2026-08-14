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


class move_cup_put_pen_in_cup(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.cup_obj.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        print_c(self.seed, "YELLOW")
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
        des_bb = get_actor_boundingbox(self.box.actor)

        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            self.obstacle_density = max(0, self.obstacle_density-1) 
            box_obs = "001_bottle"
            gap = 0.05
            place_pose =  [[des_bb[1][0]+gap if self.scene_id == 0 else des_bb[0][0]-gap, 
                           des_bb[1][1]-np.random.uniform(low=0, 
                                    high=des_bb[1][1]-des_bb[0][1]),
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

        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                         boundary_thr=[0.15, 0.2], side="left" if self.scene_id == 0 else "right")
      
        self.target_name = "058_markpen"
        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, col_thr=0.0, xlim=[0],
                    ylim=[-0.087], qpos=(90,0,0),object_bounds= object_bounds,
                     task_objs=task_objs,  obj_id = 0, mass = 0.1)
        
        self.add_prohibit_area(self.target_obj, padding=0.1, area="table")

        object_bounds.append(get_actor_boundingbox(self.target_obj.actor))

   
        self.cup_name = "021_cup"
        place_pose =  [[*self.box.get_pose().p[:2], des_bb[0][-1] + 0.03],(90,0,90)]
       
        self.cup_obj, self.cup_obj_id, self.cup_obj_pose = \
            place_actor(self.cup_name, self, task_objs = task_objs,
                    obj_pose=place_pose, mass = 0.4)
        

        if self.scene_id == 0:
            xlim = [des_bb[1][0]+0.1, des_bb[1][0]+0.2]
        else:
            xlim = [des_bb[0][0]-0.2, des_bb[0][0]-0.1]
        
        self.cup_des_pose = get_random_place_pose(xlim = xlim, ylim=[-0.1,0.1],
                                             col_thr=0.10, object_bounds=object_bounds)
        
        self.add_prohibit_area(self.cup_des_pose, padding=0.12, area="table")

        # Get the placement pose
        self.cup_des_pose = self.cup_des_pose.p.tolist() + [1,0,0,0]
        self.cup_des_pose[2] += 0.02
        self.pen_dist_threshold = 0.90

        print_c(f"Placement pose of the cup {self.cup_des_pose}", "RED")

    
    
    def pick_place_cup(self, arm_tag, pre_grasp_dist=0.1, z = 0.15, pre_dis= 0.05, dis=0.005):
        self.move(self.grasp_actor(self.cup_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))

        self.move(self.move_by_displacement(arm_tag=arm_tag,x=-z if self.scene_id else z, z=z))
        self.attach_object(self.cup_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.cup_name}/collision/base{self.cup_obj_id}.glb", str(arm_tag))
        self.move(
            self.place_actor(
                self.cup_obj,
                arm_tag=arm_tag,
                target_pose= self.cup_des_pose,
                constrain= "auto",
                pre_dis=pre_dis,
                dis=dis
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z, y=-z))

        return self.cup_obj.get_pose()
    
    def play_once(self, z = 0.1, pre_dis= 0.05, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) #("right" if self.target_obj.get_pose().p[0] > 0 else "left")

        des_pose = self.pick_place_cup(arm_tag)
        des_obj_pose = des_pose.p.tolist()+ euler2quat(0,0,np.deg2rad(90)).tolist()
        des_obj_pose[2] = self.pen_dist_threshold
        # Grasp the mouse with the selected arm
        self.move(self.grasp_actor(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(arm_tag))

        self.move(self.move_by_displacement(arm_tag=arm_tag, y=-z, z=z))
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose= des_obj_pose,
                constrain= "free",
                pre_dis=pre_dis,
                dis=dis,
                actor_axis_type="world"
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
        cup_pose = self.cup_obj.get_pose().p
        eps = 0.03
        cup_in_box = np.all((box_bb[0][:2] <= cup_pose[:2])  &  (cup_pose[:2] <= box_bb[1][:2]))

        table_bb = get_actor_boundingbox(self.table)
        cup_on_table = np.all((table_bb[0][:2] <= cup_pose[:2])  &  (cup_pose[:2] <= table_bb[1][:2]))
        cup_on_table &= (cup_pose[-1] - table_bb[1][-1]) < eps  

        pen_in_cup = np.all(abs(cup_pose[:2] - self.target_obj.get_pose().p[:2]) < np.array([eps, eps]))

        # print(cup_pose, cup_in_box, box_bb)
        return (not cup_in_box and cup_on_table and pen_in_cup
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())
