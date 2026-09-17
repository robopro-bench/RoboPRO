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
from bench_envs.utils.scene_gen_utils import place_actor
from transforms3d.euler import euler2quat
from envs.utils.rand_create_actor import rand_pose


def _robotwin_log_move():
    return os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"


class move_seal_onto_book(Study_base_task):

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj_1.get_name(), self.seal_obj.get_name()}

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        kwargs["include_collison"] = False
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        with open(os.path.join(os.environ["BENCH_ROOT"],'bench_task_config', 'task_objects.yml'), "r", encoding="utf-8") as f:
            task_objs = yaml.safe_load(f)
        xlim, ylim, self.side_to_place = get_position_limits(self.table,
                                      boundary_thr=0.20, side="right" if self.scene_id == 0 else "left")
        
        object_bounds = [get_actor_boundingbox(o) for o in self.scene_objs]        
      
          #place obstacles inside and next to the box
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
      
        bcs_bb = get_actor_boundingbox(self.bookcase)
      

        x_w = bcs_bb[1][0] - bcs_bb[0][0]
        y_l = bcs_bb[1][1] - bcs_bb[0][1]

        q = (0,180,90)
        book_x_offset = np.random.choice([x_w/5, x_w/2, x_w - x_w/5])
        p_1 = [bcs_bb[0][0] + book_x_offset, bcs_bb[0][1] + y_l/2,
               self.table.get_pose().p[-1] + 0.03]
        self.arm_side = self.side_to_place
        self.target_name = "043_book"
        
        self.target_obj_1, self.target_id_1, _ = \
        place_actor(self.target_name, self, task_objs = task_objs,obj_id = 0,
                     obj_pose=[p_1,q], mass = 0.5, rotation=False)
        
        # Place seal in the box (same as move_seal_onto_table)
        des_bb = get_actor_boundingbox(self.box.actor)
        self.seal_name = "100_seal"
        box_pose = self.box.get_pose().p
        seal_place_pose = [[box_pose[0], box_pose[1]+0.02, des_bb[0][-1] + 0.03], (90, 0, 180)]

        self.seal_obj, self.seal_id, self.seal_obj_pose = \
            place_actor(self.seal_name, self, task_objs=task_objs,
                    obj_pose=seal_place_pose, mass=0.1)

        self.lift_height = 0.2
        self.ep_lift = -0.2

        # Place book on the same side as the box so one arm can reach both seal and book
        box_x = self.box.get_pose().p[0]
        if box_x > 0:
            # Box is on the right — book should be right-of-center
            book_xlim = [0.02, min(0.12, xlim[1] - 0.05)]
        else:
            # Box is on the left — book should be left-of-center
            book_xlim = [max(-0.12, xlim[0] + 0.05), -0.02]
        book_ylim = [ylim[0] + 0.05, ylim[1] - 0.1]

        if _robotwin_log_move():
            print(f"[move_seal_onto_book] book_xlim: {book_xlim}")
            print(f"[move_seal_onto_book] book_ylim: {book_ylim}")

        target_qpos = euler2quat(*[np.deg2rad(d) for d in [15, 180, 0]])
        if target_qpos[0] < 0:
            target_qpos = -target_qpos

        max_attempts = 200
        for _attempt in range(max_attempts):
            base_pose = rand_pose(
                xlim=book_xlim,
                ylim=book_ylim,
                zlim=[0.89],
                qpos=target_qpos,
                rotate_rand=None,
            )
            if not get_collison_with_objs(object_bounds, base_pose, 0.15):
                break
        else:
            print(f"[move_seal_onto_book] WARNING: no collision-free pose after {max_attempts} attempts, using last pose")

        self.book_des_pose = base_pose
        if _robotwin_log_move():
            print(f"[move_seal_onto_book] placing book at {self.book_des_pose}")
        self.add_prohibit_area(self.target_obj_1, padding=0.12, area="table")
        self.add_prohibit_area(self.book_des_pose, padding=0.12, area="table")

      
    def play_once(self, pre_dis= 0.07, dis=0.005, pre_grasp_dist=0.1):
        # --- Step 1: Move book from bookcase to table ---
        book_x = self.target_obj_1.get_pose().p[0]
        book_arm_tag = ArmTag("right" if book_x > 0 else "left")

        if _robotwin_log_move():
            _p = self.target_obj_1.get_pose()
            print(
                f"[move_seal_onto_book] book (actor {self.target_obj_1.get_name()}): "
                f"p={np.round(np.asarray(_p.p), 4)} q={np.round(np.asarray(_p.q), 4)}"
            )

        self.move(self.grasp_actor(self.target_obj_1, arm_tag=book_arm_tag, pre_grasp_dis=pre_grasp_dist))
        self.move(self.move_by_displacement(arm_tag=book_arm_tag, y=self.ep_lift,
                                            z=self.lift_height))
        self.attach_object(self.target_obj_1, f"{os.environ['ASSETS_ROOT']}/objects/{self.target_name}/collision/base{self.target_id_1}.glb", str(book_arm_tag))
        self.add_collision()
        self.update_world()
        self.move(
            self.place_actor(
                self.target_obj_1,
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

        # --- Step 2: Move seal from box directly onto the book ---
        seal_x = self.seal_obj.get_pose().p[0]
        book_pose = self.target_obj_1.get_pose()
        seal_arm_tag = ArmTag("right" if seal_x > 0 else "left")

        if seal_arm_tag != book_arm_tag:
            self.move(self.back_to_origin(book_arm_tag))

        if _robotwin_log_move():
            _p = self.seal_obj.get_pose()
            print(
                f"[move_seal_onto_book] seal (actor {self.seal_obj.get_name()}): "
                f"p={np.round(np.asarray(_p.p), 4)} q={np.round(np.asarray(_p.q), 4)}"
            )
            print(f"[move_seal_onto_book] seal_arm={seal_arm_tag}")

       
        # Grab seal from box
        self.move(self.grasp_actor(self.seal_obj, arm_tag=seal_arm_tag, pre_grasp_dis=pre_grasp_dist))
        seal_x_disp = 0.15 if seal_x <= 0 else -0.15
        self.move(self.move_by_displacement(arm_tag=seal_arm_tag, x=seal_x_disp,
                                            z=0.15))
        self.attach_object(self.seal_obj, f"{os.environ['ASSETS_ROOT']}/objects/{self.seal_name}/collision/base{self.seal_id}.glb", str(seal_arm_tag))

        # Place seal directly on top of the book
        book_bb = get_actor_boundingbox(self.target_obj_1.actor)
        seal_des_pose = sapien.Pose(
            [book_pose.p[0], book_pose.p[1], book_bb[1][2] + 0.03],
            [1,0,0,0]
        )
        self.move(
            self.place_actor(
                self.seal_obj,
                arm_tag=seal_arm_tag,
                target_pose=seal_des_pose,
                constrain="free",
                pre_dis=0,
                dis=dis,
            ))
        self.detach_object(seal_arm_tag)
        self.move(self.move_by_displacement(arm_tag=seal_arm_tag, z=0.03))

        self.info["info"] = {
            "{A}": f"{self.seal_name}/base{self.seal_id}",
            "{B}": f"{self.target_name}/base{self.target_id_1}",
            "{a}": str(seal_arm_tag),
        }
        return self.info

    def check_success(self):
        eps = 0.05  # 5cm
        book_p = np.asarray(self.target_obj_1.get_pose().p)
        seal_p = np.asarray(self.seal_obj.get_pose().p)
        table_bb = get_actor_boundingbox(self.table)
        book_bb = get_actor_boundingbox(self.target_obj_1)

        # Book must be on the table surface (xy within table bounds)
        book_on_table = (table_bb[0][0] <= book_p[0] <= table_bb[1][0] and
                         table_bb[0][1] <= book_p[1] <= table_bb[1][1])

        # Seal must be on the book (xy within book bounds, z above book top)
        seal_on_book_xy = (book_bb[0][0] <= seal_p[0] <= book_bb[1][0] and
                           book_bb[0][1] <= seal_p[1] <= book_bb[1][1])
        dz = float(seal_p[2] - book_bb[1][2])
        seal_on_book_z = 0 <= dz <= eps

        # Seal must not be in the box
        box_bb = get_actor_boundingbox(self.box.actor)
        seal_in_box = (box_bb[0][0] <= seal_p[0] <= box_bb[1][0] and
                       box_bb[0][1] <= seal_p[1] <= box_bb[1][1])

        if _robotwin_log_move():
            print(
                f"[move_seal_onto_book] check_success "
                f"book_on_table={book_on_table} "
                f"seal_on_book_xy={seal_on_book_xy} "
                f"dz(seal-book_top)={dz:.4f} seal_on_book_z={seal_on_book_z} "
                f"seal_in_box={seal_in_box}"
            )

        return bool(book_on_table and seal_on_book_xy and seal_on_book_z and not seal_in_box)
