import sapien
import numpy as np
from copy import deepcopy
import transforms3d as t3d
from pathlib import Path

from . import transforms
from .transforms import *

from sapien import Entity
from sapien.physx import PhysxArticulation, PhysxArticulationLinkComponent

from typing import Literal, Generator


class Actor:
    POINTS = {
        "contact": "contact_points_pose",
        "target": "target_pose",
        "functional": "functional_matrix",
        "orientation": "orientation_point",
    }

    def __init__(self, actor: Entity, actor_data: dict, mass=0.01, scale=None):
        self.actor = actor
        self.config = actor_data
        self.set_mass(mass)
        self.scale = scale

    def get_point(
        self,
        type: Literal["contact", "target", "functional", "orientation"],
        idx: int,
        ret: Literal["matrix", "list", "pose"],
    ) -> np.ndarray | list | sapien.Pose:
        """Get the point of the entity actor."""
        type = self.POINTS[type]

        actor_matrix = self.actor.get_pose().to_transformation_matrix()
        try:
            local_matrix = np.array(self.config[type][idx])
        except:
            return None
        local_matrix[:3, 3] *= np.array(self.scale)

        world_matrix = actor_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3], t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        return self.actor.get_pose()

    def get_contact_point(self,
                          idx: int,
                          ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of given contact point of the actor."""
        return self.get_point("contact", idx, ret)

    def iter_contact_points(
        self,
        ret: Literal["matrix", "list", "pose"] = "list"
    ) -> Generator[tuple[int, np.ndarray | list | sapien.Pose], None, None]:
        """Iterate over all contact points of the actor."""
        for i in range(len(self.config[self.POINTS["contact"]])):
            yield i, self.get_point("contact", i, ret)

    def get_functional_point(self,
                             idx: int,
                             ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of given functional point of the actor."""
        return self.get_point("functional", idx, ret)

    def get_target_point(self,
                         idx: int,
                         ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of given target point of the actor."""
        return self.get_point("target", idx, ret)

    def get_orientation_point(self, ret: Literal["matrix", "list", "pose"] = "list") -> np.ndarray | list | sapien.Pose:
        """Get the transformation matrix of given orientation point of the actor."""
        return self.get_point("orientation", 0, ret)

    def get_name(self):
        return self.actor.get_name()

    def set_name(self, name):
        self.actor.set_name(name)

    def set_mass(self, mass):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                component.mass = mass

    def set_friction(self, static_friction=5.0, dynamic_friction=4.0, restitution=0.0):
        """Raise contact friction on this actor's collision shapes.

        Bench scene default is static=2 / dynamic=1, which is too slippery
        for gripper contact. PhysX averages the two materials, so bumping
        the object is enough even if the fingers stay at the scene default.
        """
        scene = getattr(self.actor, "scene", None)
        if scene is None:
            getter = getattr(self.actor, "get_scene", None)
            scene = getter() if getter is not None else None
        if scene is None:
            return
        mat = scene.create_physical_material(static_friction, dynamic_friction, restitution)
        for component in self.actor.get_components():
            if not isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                continue
            shapes = []
            if hasattr(component, "get_collision_shapes"):
                try:
                    shapes = list(component.get_collision_shapes() or [])
                except Exception:
                    shapes = []
            if not shapes:
                shapes = list(getattr(component, "collision_shapes", None) or [])
            for shape in shapes:
                if hasattr(shape, "set_physical_material"):
                    shape.set_physical_material(mat)
                elif hasattr(shape, "physical_material"):
                    shape.physical_material = mat


class ArticulationActor(Actor):
    POINTS = {
        "contact": "contact_points",
        "target": "target_points",
        "functional": "functional_points",
        "orientation": "orientation_point",
    }

    def __init__(self, actor: PhysxArticulation, actor_data: dict, mass=0.01, scale=None):
        assert isinstance(actor, PhysxArticulation), "ArticulationActor must be a Articulation"

        self.actor = actor
        self.config = actor_data

        self.link_dict = self.get_link_dict()
        self.set_mass(mass)
        self.scale = [scale,scale,scale] if isinstance(scale, float) else scale
        
    def get_link_dict(self) -> dict[str, PhysxArticulationLinkComponent]:
        link_dict = {}
        for link in self.actor.get_links():
            link_dict[link.get_name()] = link
        return link_dict

    def get_link_pose(self, link_name: str) -> sapien.Pose:
        """Return the world pose of the given link."""
        return self.link_dict[link_name].get_pose()

    def get_point(
        self,
        type: Literal["contact", "target", "functional", "orientation"],
        idx: int,
        ret: Literal["matrix", "list", "pose"],
    ) -> np.ndarray | list | sapien.Pose:
        """Get the point of the articulation actor."""
        type = self.POINTS[type]
        local_matrix = np.array(self.config[type][idx]["matrix"])
        local_matrix[:3, 3] *= self.config["scale"]

        link = self.link_dict[self.config[type][idx]["base"]]
        link_matrix = link.get_pose().to_transformation_matrix()

        world_matrix = link_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() + t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3], t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def set_mass(self, mass, links_name: list[str] = None):
        for link in self.actor.get_links():
            if links_name is None or link.get_name() in links_name:
                link.set_mass(mass)

    def set_properties(self, damping, stiffness, friction=None, force_limit=None):
        for joint in self.actor.get_joints():
            if force_limit is not None:
                joint.set_drive_properties(damping=damping, stiffness=stiffness, force_limit=force_limit)
            else:
                joint.set_drive_properties(
                    damping=damping,
                    stiffness=stiffness,
                )
            if friction is not None:
                joint.set_friction(friction)

    def set_qpos(self, qpos):
        self.actor.set_qpos(qpos)

    def set_qvel(self, qvel):
        self.actor.set_qvel(qvel)

    def get_qlimits(self):
        return self.actor.get_qlimits()

    def get_qpos(self):
        return self.actor.get_qpos()

    def get_qvel(self):
        return self.actor.get_qvel()
