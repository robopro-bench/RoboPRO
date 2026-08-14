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


class empty_box(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        print_c(self.seed, "YELLOW")
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
            
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                         boundary_thr=[0.10, 0.10],
                                        side="left" if self.scene_id == 0 else "right")

        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]
        # Place a bottle next to the box
        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            self.obstacle_density = max(0, self.obstacle_density-1)  # Ensure non-negative density
            des_bb = get_actor_boundingbox(self.box.actor)
            box_obs = "001_bottle"
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

   
        des_bb = get_actor_boundingbox(self.box.actor)
    
        # Object 1
        self.seal_name ="100_seal" 
        box_pose = self.box.get_pose().p
        place_pose =  [[box_pose[0]+np.random.uniform(-0.02, 0.02), box_pose[1]-0.06, des_bb[0][-1] + 0.03],(90,0,180)]
        self.seal_obj, self.seal_obj_id, self.seal_obj_pose = \
            place_actor(self.seal_name, self, task_objs = task_objs,
                    obj_pose=place_pose, mass = 0.1)

        # seal_xlim = [xlim[0] + np.mean(xlim), xlim[1]]
        seal_xlim = [xlim[0] + np.mean(xlim), xlim[1]] if self.side_to_place == "right" else [xlim[0], xlim[1]+ np.mean(xlim)]
        seal_ylim = [ylim[0], ylim[1]-0.15]
        # print_c(f"seal_xlim: {seal_xlim}, seal_ylim: {seal_ylim}", "BLUE")
        self.seal_des_pose = get_random_place_pose(xlim=seal_xlim, ylim=seal_ylim,
                                             col_thr=0.05, object_bounds=object_bounds)
        self.add_prohibit_area(self.seal_des_pose, padding=0.1, area="table")

        # Object 2
        place_pose =  [[box_pose[0]+np.random.uniform(-0.02, 0.02), box_pose[1] + 0.04, des_bb[0][-1] + 0.03],(90,0,90)]
     
        self.cup_name = "021_cup" 
        self.cup_obj, self.cup_obj_id, self.cup_obj_pose = \
            place_actor(self.cup_name, self, task_objs = task_objs,
                    obj_pose=place_pose, mass = 0.4, obj_id=2, scale=0.1)

        cup_bb = get_actor_boundingbox(self.cup_obj.actor)

        #shift bb with expected pose
        cup_curr_c = [(cup_bb[1][0] + cup_bb[0][0])/2, (cup_bb[1][1] + cup_bb[0][1])/2]
        trans_c = [self.seal_des_pose.p[0]-cup_curr_c[0], self.seal_des_pose.p[1] - cup_curr_c[1]]
        cup_bb[0][:2] += trans_c
        cup_bb[1][:2] += trans_c

        object_bounds.append(cup_bb)

        # Get the placement pose
        self.seal_des_pose = self.seal_des_pose.p.tolist() + [1,0,0,0]
        cup_xlim = [xlim[0] + np.mean(xlim), xlim[1]] if self.side_to_place == "right" else [xlim[0], xlim[1]+ np.mean(xlim)]
        cup_ylim = [ylim[0], ylim[1]-0.15]
        # print_c(f"cup_xlim: {cup_xlim}, cup_ylim: {cup_ylim}", "BLUE")
        self.cup_des_pose = get_random_place_pose(xlim=cup_xlim, ylim=cup_ylim,
                                             col_thr=0.05, object_bounds=object_bounds)
       
        self.add_prohibit_area(self.cup_des_pose, padding=0.1, area="table")

        self.cup_des_pose = self.cup_des_pose.p.tolist() + [1,0,0,0]

        # print_c(f"Placement destination poses: Seal {self.seal_des_pose}; \
        #         Cup {self.cup_des_pose}", "RED")

    def _get_target_object_names(self) -> set[str]:
        """Default for tasks with single self.target_obj. Override for multi-target tasks."""
        return {self.cup_obj.get_name(), self.seal_obj.get_name()}
    
    def pick_place_seal(self, arm_tag, pre_grasp_dist=0.1,
                        z = 0.08, pre_dis= 0.05, dis=0.005):
        self.move(self.grasp_actor(self.seal_obj, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dist))
        # self.move(self.move_by_displacement(arm_tag=arm_tag,x=z if self.side_to_place == "left" else -z, z=z,
        #                                     constraint_pose=None))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z+pre_dis))
        self.attach_object(self.seal_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.seal_name}/collision/base{self.seal_obj_id}.glb", str(arm_tag))
        self.move(
            self.place_actor(
                self.seal_obj,
                arm_tag=arm_tag,
                target_pose= self.seal_des_pose,
                constrain= "free",
                actor_axis_type="world",
                pre_dis=pre_dis,
                dis=dis
            ))
        # print(" successfully place seal, lift arm...")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z))
        # print(" successfully lift arm")

        return self.seal_obj.get_pose()
        
    def pick_place_cup(self, arm_tag, pre_grasp_dist=0.1,
                        z = 0.08, pre_dis= 0.05, dis=0.005):
        self.move(self.grasp_actor(self.cup_obj, arm_tag=arm_tag,
                                    pre_grasp_dis=pre_grasp_dist))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, x=z if self.side_to_place == "left" else -z, z=z,
        #                                     constraint_pose=None))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=z+pre_dis))
        self.attach_object(self.cup_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.cup_name}/collision/base{self.cup_obj_id}.glb", str(arm_tag))
        self.move(
            self.place_actor(
                self.cup_obj,
                arm_tag=arm_tag,
                target_pose= self.cup_des_pose,
                constrain= "free",
                actor_axis_type="world",
                pre_dis=pre_dis,
                dis=dis
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))

        return self.cup_obj.get_pose()
    
    def play_once(self, z = 0.15, pre_dis= 0.05, dis=0.005, pre_grasp_dist=0.1):
        # Determine which arm to use based on mouse position (right if on right side, left otherwise)
        arm_tag = ArmTag(self.side_to_place ) 
        self.pick_place_seal(arm_tag)
        self.pick_place_cup(arm_tag)

        # Record information about the objects and arm used in the task
    
        self.info["info"] = {
            "{A}": f"{self.cup_name}/base{self.cup_obj_id}",
            "{B}": f"red",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        box_bb = get_actor_boundingbox(self.box.actor)
        eps = 0.05
        
        obj_poses = np.stack([ob.get_pose().p[:2] for ob in [self.cup_obj, self.seal_obj]], 
                             axis=0)
        cup_in_box = np.all((box_bb[0][:2] <= self.cup_obj.get_pose().p[:2])  &  (self.cup_obj.get_pose().p[:2] <= box_bb[1][:2]))
        seal_in_box = np.all((box_bb[0][:2] <= self.seal_obj.get_pose().p[:2])  &  (self.seal_obj.get_pose().p[:2] <= box_bb[1][:2]))

        table_bb = get_actor_boundingbox(self.table)
        objects_on_table = np.all((table_bb[0][:2] <= obj_poses)  &  (obj_poses <= table_bb[1][:2]))
        objects_on_table &= (obj_poses[..., -1] - table_bb[1][-1]) < eps  
        
        return (not cup_in_box and not seal_in_box and np.all(objects_on_table)
                and self.robot.is_left_gripper_open() 
                and self.robot.is_right_gripper_open())