"""
Advanced Primitive Detection System
Detects 4 atomic primitives: Lane Following (LF), Lane Changing (LC), Yielding (Y), Turning (T)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import config


class PrimitiveDetector:
    """
    Detects driving behavior primitives from trajectory data
    """
    
    def __init__(self):
        self.config = config.PRIMITIVE_CONFIG
        
    def detect_primitives(
        self,
        trajectory: np.ndarray,
        map_data: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Detect all primitives for a given trajectory
        
        Args:
            trajectory: [timesteps, 7] where 7 = [x, y, heading, vx, vy, ax, ay]
            map_data: Optional map/lane information
        
        Returns:
            Dictionary with primitive probabilities: {'LF': prob, 'LC': prob, 'Y': prob, 'T': prob}
        """
        if trajectory.shape[0] < 2:
            return {'LF': 0.0, 'LC': 0.0, 'Y': 0.0, 'T': 0.0}
        
        # Extract features
        positions = trajectory[:, :2]  # [timesteps, 2] (x, y)
        headings = trajectory[:, 2]    # [timesteps] (heading)
        velocities = trajectory[:, 3:5]  # [timesteps, 2] (vx, vy)
        accelerations = trajectory[:, 5:7]  # [timesteps, 2] (ax, ay)
        
        # Compute derived features
        speeds = np.linalg.norm(velocities, axis=1)  # [timesteps]
        lateral_velocities = self._compute_lateral_velocity(velocities, headings)
        heading_rates = np.diff(headings)  # [timesteps-1]
        accelerations_mag = np.linalg.norm(accelerations, axis=1)  # [timesteps]
        
        # Detect each primitive
        lf_prob = self._detect_lane_following(positions, headings, speeds, map_data)
        lc_prob = self._detect_lane_changing(positions, lateral_velocities, headings, map_data)
        y_prob = self._detect_yielding(positions, speeds, accelerations, map_data)
        t_prob = self._detect_turning(positions, headings, heading_rates, map_data)
        
        return {
            'LF': float(lf_prob),
            'LC': float(lc_prob),
            'Y': float(y_prob),
            'T': float(t_prob)
        }
    
    def _detect_lane_following(
        self,
        positions: np.ndarray,
        headings: np.ndarray,
        speeds: np.ndarray,
        map_data: Optional[Dict]
    ) -> float:
        """
        Detect Lane Following: maintaining position within lane
        
        Criteria:
        - Low lateral deviation from lane center
        - Heading aligned with lane direction
        - Consistent speed
        """
        cfg = self.config['lane_following']
        
        # Compute lateral deviation (simplified - would use map data in practice)
        if map_data is not None and 'lane_center' in map_data:
            # Use actual lane center from map
            lane_center = map_data['lane_center']
            lateral_deviations = self._compute_lateral_deviation(positions, lane_center)
        else:
            # Estimate from trajectory smoothness
            # Lane following should have low curvature
            lateral_deviations = self._estimate_lateral_deviation(positions)
        
        # Check heading alignment (low heading variation)
        heading_variance = np.var(headings)
        heading_aligned = heading_variance < cfg['heading_threshold'] ** 2
        
        # Check lateral deviation
        avg_lateral_dev = np.mean(np.abs(lateral_deviations))
        low_lateral_dev = avg_lateral_dev < cfg['lateral_threshold']
        
        # Check speed consistency
        speed_variance = np.var(speeds)
        speed_consistent = speed_variance < 1.0  # m/s variance threshold
        
        # Combine evidence
        evidence = 0.0
        if low_lateral_dev:
            evidence += 0.4
        if heading_aligned:
            evidence += 0.3
        if speed_consistent:
            evidence += 0.3
        
        return min(evidence, 1.0)
    
    def _detect_lane_changing(
        self,
        positions: np.ndarray,
        lateral_velocities: np.ndarray,
        headings: np.ndarray,
        map_data: Optional[Dict]
    ) -> float:
        """
        Detect Lane Changing: lateral movement between lanes
        
        Criteria:
        - Significant lateral velocity
        - Lateral position change
        - Lane boundary crossing (if map available)
        """
        cfg = self.config['lane_changing']
        
        # Check lateral velocity
        avg_lateral_vel = np.mean(np.abs(lateral_velocities))
        high_lateral_vel = avg_lateral_vel > cfg['lateral_velocity_threshold']
        
        # Check lateral displacement
        lateral_displacement = np.abs(positions[-1, 1] - positions[0, 1])  # Assuming y is lateral
        significant_displacement = lateral_displacement > cfg['lane_crossing_threshold']
        
        # Check for lane boundary crossing (simplified)
        if map_data is not None:
            lane_crossing = self._detect_lane_boundary_crossing(positions, map_data)
        else:
            # Estimate from trajectory pattern
            lane_crossing = significant_displacement and high_lateral_vel
        
        # Combine evidence
        evidence = 0.0
        if high_lateral_vel:
            evidence += 0.4
        if significant_displacement:
            evidence += 0.3
        if lane_crossing:
            evidence += 0.3
        
        return min(evidence, 1.0)
    
    def _detect_yielding(
        self,
        positions: np.ndarray,
        speeds: np.ndarray,
        accelerations: np.ndarray,
        map_data: Optional[Dict]
    ) -> float:
        """
        Detect Yielding: decelerating for other agents
        
        Criteria:
        - Negative acceleration (deceleration)
        - Speed reduction
        - Proximity to other agents (if available)
        """
        cfg = self.config['yielding']
        
        # Check for deceleration
        avg_acceleration = np.mean(accelerations[:, 0])  # Longitudinal acceleration
        decelerating = avg_acceleration < cfg['deceleration_threshold']
        
        # Check speed reduction
        speed_reduction = speeds[-1] < speeds[0] - 0.5  # At least 0.5 m/s reduction
        
        # Check proximity to other agents (would use map_data in practice)
        proximity_detected = False
        if map_data is not None and 'other_agents' in map_data:
            proximity_detected = self._check_proximity(positions, map_data['other_agents'], cfg['proximity_threshold'])
        
        # Combine evidence
        evidence = 0.0
        if decelerating:
            evidence += 0.5
        if speed_reduction:
            evidence += 0.3
        if proximity_detected:
            evidence += 0.2
        
        return min(evidence, 1.0)
    
    def _detect_turning(
        self,
        positions: np.ndarray,
        headings: np.ndarray,
        heading_rates: np.ndarray,
        map_data: Optional[Dict]
    ) -> float:
        """
        Detect Turning: intersection navigation
        
        Criteria:
        - High heading rate of change
        - Curved trajectory
        - Intersection entry/exit (if map available)
        """
        cfg = self.config['turning']
        
        # Check heading rate
        avg_heading_rate = np.mean(np.abs(heading_rates))
        high_heading_rate = avg_heading_rate > cfg['heading_rate_threshold']
        
        # Check curvature
        curvature = self._compute_curvature(positions)
        high_curvature = np.mean(curvature) > cfg['curvature_threshold']
        
        # Check for intersection (simplified)
        in_intersection = False
        if map_data is not None and 'intersections' in map_data:
            in_intersection = self._check_intersection(positions, map_data['intersections'])
        else:
            # Estimate from trajectory pattern
            in_intersection = high_curvature and high_heading_rate
        
        # Combine evidence
        evidence = 0.0
        if high_heading_rate:
            evidence += 0.4
        if high_curvature:
            evidence += 0.3
        if in_intersection:
            evidence += 0.3
        
        return min(evidence, 1.0)
    
    # Helper methods
    
    def _compute_lateral_velocity(self, velocities: np.ndarray, headings: np.ndarray) -> np.ndarray:
        """Compute lateral velocity component"""
        # Rotate velocities to lateral direction (perpendicular to heading)
        lateral_velocities = np.zeros(len(velocities))
        for i in range(len(velocities)):
            # Lateral is perpendicular to heading
            lateral_dir = np.array([-np.sin(headings[i]), np.cos(headings[i])])
            lateral_velocities[i] = np.dot(velocities[i], lateral_dir)
        return lateral_velocities
    
    def _compute_lateral_deviation(self, positions: np.ndarray, lane_center: np.ndarray) -> np.ndarray:
        """Compute lateral deviation from lane center"""
        # Simplified: distance to lane center line
        deviations = np.zeros(len(positions))
        for i in range(len(positions)):
            # Find closest point on lane center
            dists = np.linalg.norm(lane_center - positions[i], axis=1)
            deviations[i] = np.min(dists)
        return deviations
    
    def _estimate_lateral_deviation(self, positions: np.ndarray) -> np.ndarray:
        """Estimate lateral deviation from trajectory smoothness"""
        # Fit a line to positions and compute deviations
        if len(positions) < 3:
            return np.zeros(len(positions))
        
        # Use first and last point to define expected path
        start = positions[0]
        end = positions[-1]
        direction = end - start
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            return np.zeros(len(positions))
        
        direction = direction / direction_norm
        
        # Compute perpendicular distance for each point
        deviations = np.zeros(len(positions))
        for i, pos in enumerate(positions):
            vec = pos - start
            proj_length = np.dot(vec, direction)
            proj_point = start + proj_length * direction
            deviations[i] = np.linalg.norm(pos - proj_point)
        
        return deviations
    
    def _detect_lane_boundary_crossing(self, positions: np.ndarray, map_data: Dict) -> bool:
        """Detect if trajectory crosses lane boundaries"""
        # Simplified: would use actual lane boundaries from map
        return False  # Placeholder
    
    def _check_proximity(self, positions: np.ndarray, other_agents: np.ndarray, threshold: float) -> bool:
        """Check if trajectory is near other agents"""
        for pos in positions:
            for agent_pos in other_agents:
                dist = np.linalg.norm(pos - agent_pos)
                if dist < threshold:
                    return True
        return False
    
    def _compute_curvature(self, positions: np.ndarray) -> np.ndarray:
        """Compute trajectory curvature"""
        if len(positions) < 3:
            return np.zeros(len(positions))
        
        curvatures = np.zeros(len(positions))
        for i in range(1, len(positions) - 1):
            p1 = positions[i-1]
            p2 = positions[i]
            p3 = positions[i+1]
            
            # Compute curvature using three points
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product magnitude
            cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 1e-6 and norm2 > 1e-6:
                curvatures[i] = cross / (norm1 * norm2 * (norm1 + norm2))
            else:
                curvatures[i] = 0.0
        
        # Copy edge values
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures
    
    def _check_intersection(self, positions: np.ndarray, intersections: List) -> bool:
        """Check if trajectory passes through intersection"""
        # Simplified: would use actual intersection polygons
        return False  # Placeholder
    
    def detect_primitive_combination(self, primitive_probs: Dict[str, float], threshold: float = 0.3) -> List[str]:
        """
        Detect which primitives are active (multi-primitive combinations)
        
        Args:
            primitive_probs: Dictionary of primitive probabilities
            threshold: Minimum probability to consider primitive active
        
        Returns:
            List of active primitive names
        """
        active = []
        for name, prob in primitive_probs.items():
            if prob >= threshold:
                active.append(name)
        return active
    
    def get_primitive_one_hot(self, primitive_probs: Dict[str, float], threshold: float = 0.3) -> torch.Tensor:
        """
        Convert primitive probabilities to one-hot encoding
        
        Args:
            primitive_probs: Dictionary of primitive probabilities
            threshold: Minimum probability to consider primitive active
        
        Returns:
            One-hot tensor [4] for [LF, LC, Y, T]
        """
        active = self.detect_primitive_combination(primitive_probs, threshold)
        one_hot = torch.zeros(4)
        primitive_map = {'LF': 0, 'LC': 1, 'Y': 2, 'T': 3}
        for name in active:
            if name in primitive_map:
                one_hot[primitive_map[name]] = 1.0
        return one_hot


def test_primitive_detection():
    """Test primitive detection on sample trajectories"""
    detector = PrimitiveDetector()
    
    # Test 1: Lane Following
    print("Test 1: Lane Following")
    lf_traj = np.array([
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
    ])
    probs = detector.detect_primitives(lf_traj)
    print(f"  Primitives: {probs}")
    print(f"  Active: {detector.detect_primitive_combination(probs)}")
    
    # Test 2: Lane Changing
    print("\nTest 2: Lane Changing")
    lc_traj = np.array([
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 10.0, 0.5, 0.0, 0.0],
        [2.0, 1.0, 0.0, 10.0, 0.5, 0.0, 0.0],
        [3.0, 1.5, 0.0, 10.0, 0.0, 0.0, 0.0],
    ])
    probs = detector.detect_primitives(lc_traj)
    print(f"  Primitives: {probs}")
    print(f"  Active: {detector.detect_primitive_combination(probs)}")
    
    # Test 3: Yielding
    print("\nTest 3: Yielding")
    y_traj = np.array([
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 8.0, 0.0, -2.0, 0.0],
        [2.0, 0.0, 0.0, 6.0, 0.0, -2.0, 0.0],
        [3.0, 0.0, 0.0, 4.0, 0.0, -2.0, 0.0],
    ])
    probs = detector.detect_primitives(y_traj)
    print(f"  Primitives: {probs}")
    print(f"  Active: {detector.detect_primitive_combination(probs)}")
    
    # Test 4: Turning
    print("\nTest 4: Turning")
    t_traj = np.array([
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        [0.7, 0.7, 0.785, 7.07, 7.07, 0.0, 0.0],
        [1.0, 1.0, 1.57, 0.0, 10.0, 0.0, 0.0],
        [0.7, 1.7, 2.356, -7.07, 7.07, 0.0, 0.0],
    ])
    probs = detector.detect_primitives(t_traj)
    print(f"  Primitives: {probs}")
    print(f"  Active: {detector.detect_primitive_combination(probs)}")
    
    print("\nPrimitive detection test completed!")


if __name__ == "__main__":
    test_primitive_detection()

