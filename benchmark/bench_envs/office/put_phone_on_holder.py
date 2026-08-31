# Demo for placing target_obj on des_obj on shelf.
# from envs._base_task import Base_Task
from bench_envs.office._office_base_task import Office_base_task
from envs.utils import *
import sapien
import math
from envs._GLOBAL_CONFIGS import *
from copy import deepcopy
import glob


class put_phone_on_holder(Office_base_task):

    supports_eval_env = True

    def setup_demo(self, is_test=False, **kwargs):
        kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
        super()._init_task_env_(**kwargs)

    def _get_target_object_names(self) -> set[str]:
        return {self.target_obj.get_name(), self.des_obj.get_name()}

    def load_actors(self):
        self.bottle = None
        shelf_level = 0
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]

        self.phone_id = np.random.choice(self._target_ids("office", "077_phone"))
        phone_pose = rand_pose(
            xlim = [0,0.1],
            # xlim = [0,0.35],
            ylim=[-0.2,0.08],
            # ylim=[-0.23, 0.05],
            zlim=[self.office_info["table_height"]+0.01],
            qpos=ori_quat[self.phone_id],
            rotate_rand=True,
            rotate_lim=[0, 0.7, 0],
        )
        self.target_obj = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
            is_static=False,
        )
        self.target_obj.set_mass(0.01)

        stand_pose = rand_pose(
                xlim = [self.target_obj.get_pose().p[0]+0.25,0.55],
                # ylim=[0.05],
                ylim=[-0.15,0.08],
                qpos=[0.7071, 0.7071, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 6, 0],
            )

        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.des_obj = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=False,
        )
        self.des_obj.set_mass(2)
        self.collision_list.append({
            "actor": self.des_obj,
            "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/078_phonestand/collision/base{self.stand_id}.glb",
        })
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.01, area=f"table")
        if np.random.rand() > self.clean_background_rate and self.obstacle_density >0 and self.cluttered_table:
            self.obstacle_density = max(0, self.obstacle_density-1)
            #  ---------------------------------------------------------------------
            self.id_list = [i for i in range(20)]
            self.bottle_id = np.random.choice(self.id_list)
            center_x = (self.target_obj.get_pose().p[0] + self.des_obj.get_pose().p[0]) / 2
            center_y = (self.target_obj.get_pose().p[1] + self.des_obj.get_pose().p[1]) / 2
            self.bottle = rand_create_actor(
                self,
                xlim=[center_x-0.01,center_x+0.01],
                ylim=[center_y-0.03,center_y+0.03],
                modelname="001_bottle",
                rotate_rand=True,
                rotate_lim=[0, 1, 0],
                qpos=[0.66, 0.66, -0.25, -0.25],
                convex=True,
                model_id=self.bottle_id,
                scale = [0.14, 0.14, 0.14],
            )
            
            self.stabilize_object(self.bottle)
            self.add_prohibit_area(self.bottle, padding=-0.02, area="table")
            self.collision_list.append({
                "actor": self.bottle,
                "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/001_bottle/collision/base{self.bottle_id}.glb",
            })

    def capture_task_spec(self) -> dict:
        spec = {
            "phone_id": int(self.phone_id),
            "stand_id": int(self.stand_id),
            "target_name": self.target_obj.get_name(),
            "des_name": self.des_obj.get_name(),
            "bottle_id": None,
            "bottle_name": None,
        }
        if getattr(self, "bottle", None) is not None:
            spec["bottle_id"] = int(self.bottle_id)
            spec["bottle_name"] = self.bottle.get_name()
        return spec

    def load_actors_from_spec(self, spec, init_state):
        self.phone_id = int(spec["phone_id"])
        phone_pose = self.pose_from_init_state(init_state, spec.get("target_name", "077_phone"))
        self.target_obj = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
            is_static=False,
        )
        self.target_obj.set_mass(0.01)

        self.stand_id = int(spec["stand_id"])
        stand_pose = self.pose_from_init_state(init_state, spec.get("des_name", "078_phonestand"))
        self.des_obj = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=False,
        )
        self.des_obj.set_mass(2)
        self.collision_list.append({
            "actor": self.des_obj,
            "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/078_phonestand/collision/base{self.stand_id}.glb",
        })
        self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
        self.add_prohibit_area(self.des_obj, padding=0.01, area="table")

        self.bottle = None
        if spec.get("bottle_id") is not None:
            self.bottle_id = int(spec["bottle_id"])
            bottle_name = spec.get("bottle_name") or "001_bottle"
            bottle_pose = self.pose_from_init_state(init_state, bottle_name)
            self.bottle = create_actor(
                scene=self,
                pose=bottle_pose,
                modelname="001_bottle",
                convex=True,
                model_id=self.bottle_id,
                scale=[0.14, 0.14, 0.14],
                is_static=False,
            )
            self.stabilize_object(self.bottle)
            self.add_prohibit_area(self.bottle, padding=-0.02, area="table")
            self.collision_list.append({
                "actor": self.bottle,
                "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/001_bottle/collision/base{self.bottle_id}.glb",
            })

    def play_once(self):
        # Determine which arm to use based on target_obj's position (left if target_obj is on left side, else right)
        arm_tag = ArmTag("right")

        # Grasp the target_obj with specified arm
        self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.08)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.02))
        
        self.attach_object(self.target_obj, f"{os.environ['BENCH_ROOT']}/assets/objects/077_phone/collision/base{self.phone_id}.glb", str(arm_tag))
        self.enable_table(enable=True)

        # Get des_obj's functional point as target for placement
        stand_func_pose = self.des_obj.get_functional_point(0)

        # Place the target_obj onto the des_obj's functional point with alignment constraint
        self.move(
            self.place_actor(
                self.target_obj,
                arm_tag=arm_tag,
                target_pose=stand_func_pose,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0,
                constrain="align",
            ))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.4))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.4))


        # self.info["info"] = {
        #     "{A}": f"077_phone/base{self.phone_id}",
        #     "{B}": f"078_phonestand/base{self.stand_id}",
        #     "{a}": str(arm_tag),
        # }
        # return self.info

    def check_success(self):
        end_pose_actual = np.array(self.target_obj.get_pose().p)
        end_pose_desired = np.array(self.des_obj.get_functional_point(0)[:3])
        end_pose_desired[1] -= 0.01
        end_pose_desired[2] += 0.05
        eps = np.array([0.045, 0.05, 0.04])
        return (np.all(np.abs(end_pose_actual - end_pose_desired)[:3] < eps) and self.is_left_gripper_open()
                and self.is_right_gripper_open())
