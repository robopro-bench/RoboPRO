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
from bench_envs.utils.scene_gen_utils import get_random_place_pose, get_actor_boundingbox
from bench_envs.utils.scene_gen_utils import print_c, place_actor, get_position_limits
from transforms3d.euler import euler2quat

class move_book_onto_table(Study_base_task):

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        kwargs["include_collison"] = False
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]

        bcs_bb = get_actor_boundingbox(self.bookcase)
        x_w = bcs_bb[1][0] - bcs_bb[0][0]
        y_l = bcs_bb[1][1] - bcs_bb[0][1]
        delta_choice = ((x_w/5, "left"), (x_w/2, "right"), (x_w - x_w/5, "right"))
        delta_idx =  np.random.choice(3)
        delta = delta_choice[delta_idx]
        p = [bcs_bb[0][0]+ delta[0], bcs_bb[0][1] + y_l/2, self.table.get_pose().p[-1] + 0.03]
        q = (0,180,90)
       
        if self.scene_id < 2:
            self.arm_side = "right" if self.scene_id == 0 else "left"
        else:
            self.arm_side = "left" if delta[1] == 0 else "right"

        self.target_name = "043_book"
        
        self.target_obj, self.target_id, self.target_pose = \
        place_actor(self.target_name, self, task_objs = task_objs,obj_id = 0,
                     obj_pose=[p,q], mass = 0.5, rotation=False)
       
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                      boundary_thr=0.05, side=self.arm_side)
        
        # Place book on the same side as the box so one arm can reach both seal and book
        box_x = self.box.get_pose().p[0]
        if box_x > 0:
            # Box is on the right — book should be right-of-center
            book_xlim = [0.02, min(0.12, xlim[1] - 0.05)]
        else:
            # Box is on the left — book should be left-of-center
            book_xlim = [max(-0.12, xlim[0] + 0.05), -0.02]
        book_ylim = [ylim[0] + 0.05, ylim[1] - 0.1]
        target_qpos = euler2quat(*[np.deg2rad(d) for d in [15, 180, 0]])
        if target_qpos[0] < 0:
            target_qpos = -target_qpos
        self.book_des_pose = get_random_place_pose(xlim = book_xlim, ylim=book_ylim, zlim=[0.89],
                                             col_thr=0.15, qpos=target_qpos, euler=False,
                                             object_bounds=object_bounds)
        self.lift_height = 0.2
        xy_thr = 0.2
        self.ep_lift = -xy_thr 
        # print_c(f"Lifting by {self.lift_height}", "RED")
        self.add_prohibit_area(self.target_obj, padding=0.12, area="table")
        self.add_prohibit_area(self.book_des_pose, padding=0.12, area="table")
        print_c(f"Place book on {self.book_des_pose}", "RED")

    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # --- Step 1: Move book from bookcase to table ---
        book_x = self.target_obj.get_pose().p[0]
        book_arm_tag = ArmTag("right" if book_x > 0 else "left")

        self.move(self.grasp_actor(self.target_obj, arm_tag=book_arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.move(self.move_by_displacement(arm_tag=book_arm_tag, y=self.ep_lift,
                                            z=self.lift_height))
        self.attach_object(self.target_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id}.glb", str(book_arm_tag))
        self.add_collision()
        self.update_world()
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=book_arm_tag,
                target_pose=self.book_des_pose,
                constrain="free",
                actor_axis_type="world",
                pre_dis = 0,
                dis=dis,
                align_axis=None,
                is_open=False,
            ))
        
        self.move(self.move_by_displacement(arm_tag=book_arm_tag, z=-0.03))
        self.move((book_arm_tag, [Action(book_arm_tag, "open", target_gripper_pos=1.0)]))
        self.detach_object(book_arm_tag)
        self.move(self.move_by_displacement(arm_tag=book_arm_tag,
                                            y=-0.15, z=0.01))
    def check_success(self):
        eps = 0.05
        b_pose = self.target_obj.get_pose().p
        table_bb = get_actor_boundingbox(self.table)
        book_on_table = np.all((table_bb[0][:2] <= b_pose[:2])  &  (b_pose[:2] <= table_bb[1][:2]))
        book_on_table &= (b_pose[-1] - table_bb[1][-1]) < eps  
        return (book_on_table 
                and self.robot.is_left_gripper_open()
                and self.robot.is_right_gripper_open())

