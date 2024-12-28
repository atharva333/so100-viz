import rerun as rr
import numpy as np
from typing import Dict, Optional
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
import transforms3d as tf3d
import time
from stl import mesh


@dataclass
class Joint:
    name: str
    type: str
    origin: np.ndarray
    axis: np.ndarray
    parent: str
    child: str
    current_value: float = 0.0


@dataclass
class Link:
    name: str
    visual_mesh: Optional[str] = None
    collision_mesh: Optional[str] = None


class RobotVisualizer:
    def __init__(self, urdf_path: str):
        self.joints: Dict[str, Joint] = {}
        self.links: Dict[str, Link] = {}
        self.urdf_path = urdf_path
        self.parse_urdf(urdf_path)

        # Initialize rerun
        rr.init("Robot Visualizer", spawn=True)

        # Set up the 3D view
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_X_UP)

    def parse_urdf(self, urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Parse links
        for link in root.findall("link"):
            name = link.get("name")
            visual_mesh = None

            # Get visual mesh if exists
            visual = link.find("visual")
            if visual is not None:
                geometry = visual.find("geometry")
                if geometry is not None:
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        visual_mesh = mesh.get("filename")

            self.links[name] = Link(name, visual_mesh)

        # Parse joints
        for joint in root.findall("joint"):
            name = joint.get("name")
            joint_type = joint.get("type")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")

            # Parse origin
            origin = np.eye(4)
            origin_elem = joint.find("origin")
            if origin_elem is not None:
                xyz = [float(x) for x in origin_elem.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in origin_elem.get("rpy", "0 0 0").split()]
                # Convert RPY to transformation matrix
                rot_mat = tf3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], "sxyz")
                origin[:3, :3] = rot_mat
                origin[:3, 3] = xyz

            # Parse axis
            axis = np.array([1.0, 0.0, 0.0])  # default x-axis
            axis_elem = joint.find("axis")
            if axis_elem is not None:
                axis = np.array([float(x) for x in axis_elem.get("xyz").split()])

            self.joints[name] = Joint(name, joint_type, origin, axis, parent, child)

    def calculate_transform(self, joint: Joint) -> np.ndarray:
        """Calculate the transformation matrix for a joint given its current value."""
        if joint.type == "fixed":
            return joint.origin

        # For revolute joints
        if joint.type == "revolute" or joint.type == "continuous":
            rotation = tf3d.axangles.axangle2mat(joint.axis, joint.current_value)
            transform = np.eye(4)
            transform[:3, :3] = rotation
            return joint.origin @ transform

        # For prismatic joints
        elif joint.type == "prismatic":
            transform = np.eye(4)
            transform[:3, 3] = joint.axis * joint.current_value
            return joint.origin @ transform

        return joint.origin

    def compute_forward_kinematics(self, base_link: str = "Base") -> Dict[str, np.ndarray]:
        """Compute forward kinematics starting from the base link."""
        link_transforms = {base_link: np.eye(4)}

        # Find all connected joints and links
        def process_children(parent_link: str, parent_transform: np.ndarray):
            for joint in self.joints.values():
                if joint.parent == parent_link:
                    # Calculate joint transform
                    joint_transform = self.calculate_transform(joint)
                    # Calculate child link transform
                    child_transform = parent_transform @ joint_transform
                    link_transforms[joint.child] = child_transform
                    # Process this link's children
                    process_children(joint.child, child_transform)

        process_children(base_link, link_transforms[base_link])
        return link_transforms

    def visualize_robot_state(self):
        """Visualize the current robot state using rerun."""
        # Compute forward kinematics
        transforms = self.compute_forward_kinematics()

        # Log coordinate frames for each link
        for link_name, transform in transforms.items():
            position = transform[:3, 3]
            rotation = transform[:3, :3]

            # Convert rotation matrix to quaternion
            quat = tf3d.quaternions.mat2quat(rotation)

            # Log the transform
            rr.log(
                f"robot/{link_name}",
                rr.Transform3D(
                    translation=position, rotation=rr.Quaternion(xyzw=np.roll(quat, -1))  # rerun expects wxyz
                ),
                static=True,
            )

            # If there's a mesh associated with this link, log it
            if self.links[link_name].visual_mesh:
                mesh_path = Path(self.urdf_path).parent / self.links[link_name].visual_mesh
                if mesh_path.exists():
                    stl_mesh = mesh.Mesh.from_file(mesh_path.as_posix())

                    vertices = stl_mesh.vectors.reshape(-1, 3)
                    indices = np.arange(len(vertices)).reshape(-1, 3)

                    rr.log(
                        f"robot/{link_name}/mesh",
                        rr.Mesh3D(vertex_positions=vertices, triangle_indices=indices),
                        static=True,
                    )

            # Log coordinate axes for debugging
            rr.log(
                f"robot/{link_name}/axes",
                rr.Arrows3D(
                    vectors=np.eye(3) * 0.1,
                    origins=np.zeros((3, 3)),
                    colors=[[1, 0, 0, 1.0], [0, 1, 0, 1.0], [0, 0, 1, 1.0]],
                ),
                static=True,
            )

        rr.reset_time()

    def update_joint(self, joint_name: str, value: float):
        """Update a joint's value and visualize the new robot state."""
        if joint_name in self.joints:
            self.joints[joint_name].current_value = value
            self.visualize_robot_state()


def main():
    visualizer = RobotVisualizer("URDF/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf")
    visualizer.visualize_robot_state()

    # Rotate each joint continuously between 0 and 2pi
    while True:
        for joint in visualizer.joints.values():
            if joint.type == "continuous":
                for angle in np.linspace(0, 2 * np.pi, 100):
                    visualizer.update_joint(joint.name, angle)
                    time.sleep(0.05)


if __name__ == "__main__":
    main()
