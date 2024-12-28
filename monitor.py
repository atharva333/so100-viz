import threading
import time
import yaml
import numpy as np

from visualise import RobotVisualizer
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode


class RobotMonitor:
    def __init__(self, robot_interface: FeetechMotorsBus, visualizer: RobotVisualizer, calibration: dict):
        self.robot_interface = robot_interface
        self.visualizer = visualizer
        self.calibration = calibration

        self.current_position = np.array([0, 0, 0, 0, 0, 0])
        self.position_lock = threading.Lock()  # For thread-safe position updates
        self.running = True

    def monitor_robot(self):
        while self.running:
            # Simulate getting robot position
            new_position = self.get_robot_position()

            # Thread-safe update of current position
            with self.position_lock:
                self.current_position = new_position

            time.sleep(0.05)  # Run 20Hz

    def visualize(self):
        while self.running:
            # Thread-safe reading of current position
            with self.position_lock:
                position = self.current_position

            if position is not None:
                self.update_visualization(position)
            time.sleep(0.05)  # Run 20Hz

    def _get_calibrated_angles(self, values):
        # Apply difference from URDF zero to calibrated zero
        values[1] += 270
        values[2] -= 90
        values[3] -= 90
        values[4] -= 90

        # FIXME: temporary solution for inverted joint drive mode
        values[1] = 360 - values[1]
        values[4] = 360 - values[4]

        return values

    def get_robot_position(self):
        # Read latest joint angles and offset using URDF fix
        joint_angles = self.robot_interface.read("Present_Position")
        offset_angles = self._get_calibrated_angles(joint_angles)
        return np.deg2rad(offset_angles)

    def update_visualization(self, position):
        # Replace with actual visualization code
        for idx, joint in enumerate(self.visualizer.joints.values()):
            if joint.type == "continuous":
                self.visualizer.update_joint(joint.name, position[idx])

    def run(self):
        monitor_thread = threading.Thread(target=self.monitor_robot, daemon=True)
        visual_thread = threading.Thread(target=self.visualize, daemon=True)

        monitor_thread.start()
        visual_thread.start()

        try:
            monitor_thread.join()
            visual_thread.join()
        except KeyboardInterrupt:
            self.running = False
            monitor_thread.join()
            visual_thread.join()

def main():

    # Load robot interface and calibration
    cfg = yaml.full_load(open("config.yaml"))
    calibration = yaml.full_load(open("calibration/main_follower.json"))

    motors = FeetechMotorsBus(
        port=cfg["follower_arms"]["main"]["port"], motors=cfg["follower_arms"]["main"]["motors"]
    )

    motors.connect()
    motors.set_calibration(calibration)
    motors.write("Torque_Enable", TorqueMode.DISABLED.value)

    # Load visualizer that converts from joint angles to fk to visualisation logs in rerun
    visualizer = RobotVisualizer("URDF/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf")

    monitor = RobotMonitor(robot_interface=motors, visualizer=visualizer, calibration=calibration)
    monitor.run()

if __name__ == "__main__":
    main()
