import sapien
from sapien import Entity

class Simple_Actor:
    def __init__(self, actor: Entity, mass=0.01, scale=None):
        self.actor = actor
        self.scale = scale
        self.set_mass(mass)
    
    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        return self.actor.get_pose()

    def get_name(self):
        return self.actor.get_name()

    def set_name(self, name):
        self.actor.set_name(name)

    def set_mass(self, mass):
        for component in self.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                component.mass = mass

    def set_friction(self, static_friction=5.0, dynamic_friction=4.0, restitution=0.0):
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