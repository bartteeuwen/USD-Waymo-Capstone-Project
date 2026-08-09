import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

def load_and_extract_waymo_scenarios(gcs_bucket_path: str, num_shards: int = 100):
    """
    Streams TFRecord shards from GCS, parses scenario protobufs, extracts 
    map friction points, agent counts, kinematic features, and spatial nodes.
    """
    large_scale_shards = [
        f"{gcs_bucket_path}/training_20s.tfrecord-{i:05d}-of-{num_shards:05d}" 
        for i in range(num_shards)
    ]
    streaming_dataset = tf.data.TFRecordDataset(large_scale_shards, compression_type='')

    raw_extracted_scenes = []
    gnn_extracted_scenes = []

    print(f"Streaming {num_shards} shards. Extracting Map Friction Points & Kinematics...")

    for i, raw_record in enumerate(streaming_dataset):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(raw_record.numpy()))

        # Map friction tracker
        lane_count, stop_sign_count, crosswalk_count, speed_bump_count = 0, 0, 0, 0
        for mf in scenario.map_features:
            if mf.HasField('lane'): lane_count += 1
            elif mf.HasField('stop_sign'): stop_sign_count += 1
            elif mf.HasField('crosswalk'): crosswalk_count += 1
            elif mf.HasField('speed_bump'): speed_bump_count += 1

        # Object tracking counts
        vehicle_count, pedestrian_count, cyclist_count = 0, 0, 0
        max_velocity_scene, max_decel_scene, max_yaw_rate_scene = 0.0, 0.0, 0.0
        is_anomaly = False

        scenario_nodes = []

        for track in scenario.tracks:
            if track.object_type == 1: vehicle_count += 1
            elif track.object_type == 2: pedestrian_count += 1
            elif track.object_type == 3: cyclist_count += 1

            agent_max_vel = 0.0
            agent_x, agent_y = 0.0, 0.0
            prev_velocity, prev_heading = None, None

            for state in track.states:
                if not state.valid: continue

                current_velocity = math.hypot(state.velocity_x, state.velocity_y)
                if current_velocity > 38.0: is_anomaly = True
                if current_velocity > max_velocity_scene: max_velocity_scene = current_velocity

                if prev_velocity is not None and prev_heading is not None:
                    decel = prev_velocity - current_velocity
                    if decel > max_decel_scene: max_decel_scene = decel
                    yaw_rate = abs(state.heading - prev_heading)
                    if yaw_rate > max_yaw_rate_scene: max_yaw_rate_scene = yaw_rate

                if current_velocity > agent_max_vel:
                    agent_max_vel = current_velocity
                    agent_x = state.center_x
                    agent_y = state.center_y

                prev_velocity, prev_heading = current_velocity, state.heading

            if agent_max_vel > 0:
                scenario_nodes.append({
                    'agent_type': track.object_type,
                    'velocity': round(agent_max_vel, 2),
                    'pos_x': round(agent_x, 2),
                    'pos_y': round(agent_y, 2)
                })

        raw_extracted_scenes.append({
            'scenario_id': scenario.scenario_id,
            'vehicle_count': vehicle_count,
            'pedestrian_count': pedestrian_count,
            'cyclist_count': cyclist_count,
            'lane_count': lane_count,
            'stop_sign_count': stop_sign_count,
            'crosswalk_count': crosswalk_count,
            'speed_bump_count': speed_bump_count,
            'max_velocity_mps': round(max_velocity_scene, 2),
            'max_deceleration': round(max_decel_scene, 2),
            'max_yaw_rate': round(max_yaw_rate_scene, 2),
            'is_valid_physics': not is_anomaly
        })

        gnn_extracted_scenes.append({
            'scenario_id': scenario.scenario_id,
            'agents': scenario_nodes,
            'roadgraph_features_count': len(scenario.map_features)
        })

    print(f"Extraction Complete! Total Extracted Scenes: {len(raw_extracted_scenes)}")
    return pd.DataFrame(raw_extracted_scenes), gnn_extracted_scenes
