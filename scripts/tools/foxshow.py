import os
import logging
import foxglove
from yourdfpy import URDF
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from foxglove.schemas import FrameTransforms,FrameTransform,Vector3,Quaternion,Timestamp

WORLD_FRAME_ID = "world"
GROUND_FRAME_ID = "ground"
BASE_FRAME_ID = "pelvis"

# 数据保存目录
FOXSHOW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mujoco", "foxshow_data")

class FoxShow:
    def __init__(self, _urdf: str, ground_z: float = 0.0):
        foxglove.set_log_level(logging.INFO)
        print(f"Loading URDF from {_urdf} ...")
        self.robot = URDF.load(_urdf)
        self.actuated_joint_names = self.robot.actuated_joint_names
        # ground frame offset in world coordinates (meters)
        self.ground_z = float(ground_z)
        self.server = foxglove.start_server()
        # 确保保存目录存在
        os.makedirs(FOXSHOW_DATA_DIR, exist_ok=True)
        fname = os.path.join(FOXSHOW_DATA_DIR, datetime.now().strftime("e1_%y%m%d_%H%M%S") + ".mcap")
        self.sink = foxglove.open_mcap(fname)
        print(f"数据将保存到: {fname}")

    def get_joint_names(self):
        joint_names = {}
        for joint in self.actuated_joint_names:
            joint_names[joint] = 0.0
        print(f"Available joints: {list(joint_names.keys())}")
        return list(joint_names.keys())

    def update_robot_state(self, tic = None, pos = None, vel = None, trans=None, quat = None):
        self.robot.update_cfg(pos)
        if tic is not None:
            sec = tic // 1000
            nsec = (tic % 1000) * 1000000
            stamp = Timestamp(sec=sec, nsec=nsec)
        else :
            stamp = Timestamp.now()
        if trans is None:
            trans = [0.0, 0.0, 0.742]
        self.__update_tf(stamp, trans, quat, pos, vel)

    def update_robot_target(self, tic = None, pos = None):
        if tic is not None:
            sec = tic // 1000
            nsec = (tic % 1000) * 1000000
            stamp = Timestamp(sec=sec, nsec=nsec)
        else :
            stamp = Timestamp.now()
        foxglove.log("/joint_target", {"timestamp": stamp.sec * 1_000_000_000 + stamp.nsec, "frame_id": "pelvis",
        "name": self.actuated_joint_names, "position": pos.tolist()})

    def __update_tf(self, _stamp, _trans, _quat, _pos, _vel):
        transforms = [
            FrameTransform(
                timestamp=_stamp,
                parent_frame_id=WORLD_FRAME_ID,
                child_frame_id=GROUND_FRAME_ID,
                translation=Vector3(x=0.0, y=0.0, z=self.ground_z),
                rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            ),
            FrameTransform(
                timestamp=_stamp,
                parent_frame_id=WORLD_FRAME_ID,
                child_frame_id=BASE_FRAME_ID,
                translation=Vector3(x=_trans[0], y=_trans[1], z=_trans[2]),
                rotation=Quaternion(x=_quat[0], y=_quat[1], z=_quat[2], w=_quat[3])
            ),
        ]
        # World -> Base
        # Per-joint transforms
        for joint in self.robot.robot.joints:
            parent_link = joint.parent
            child_link = joint.child
            T_local = self.robot.get_transform(frame_to=child_link, frame_from=parent_link)
            trans = T_local[:3, 3]
            quat = R.from_matrix(T_local[:3, :3]).as_quat()
            transforms.append(
                FrameTransform(
                    timestamp=_stamp,
                    parent_frame_id=parent_link,
                    child_frame_id=child_link,
                    translation=Vector3(x=float(trans[0]), y=float(trans[1]), z=float(trans[2])),
                    rotation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
                )
            )

        foxglove.log("/tf",FrameTransforms(transforms=transforms))
        foxglove.log("/joint_states", {"timestamp": _stamp.sec * 1_000_000_000 + _stamp.nsec, "frame_id": "pelvis",
        "name": self.actuated_joint_names, "position": _pos.tolist(), "velocity": _vel.tolist()})

    def stop_server(self):
        self.server.stop()
        self.sink.close()
        print("Shutting down Foxglove viewer...")
