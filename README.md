# Robot Navigation and Point Cloud Alignment

This repository contains two core algorithms widely used in robotics and computer vision:

1. **Extended Kalman Filter (EKF):** Used for localizing a robot in 2D.  
2. **Iterative Closest Point (ICP):** Used to align 3D point clouds.
3. **Route planning in occupancy grids with A* search:** 
4. **Route planning with probabilistic roadmaps**

---

## Features

### Extended Kalman Filter (EKF)
- Simulates robot motion with noisy measurements.
- Estimates the robot’s position over time.  
- Visualizes true and estimated paths with confidence bounds.  

### Iterative Closest Point (ICP)
- Aligns two 3D point clouds using rigid transformations.  
- Computes optimal rotation and translation.  
- Visualizes alignment results.

### Route Planning Using Occupancy Grid Maps with A*

- Implements A Search* for pathfinding on an occupancy grid map, leveraging a priority queue for efficient exploration.
- Uses Euclidean distance as the heuristic to estimate the cost from each node to the goal.
- Ensures accurate navigation by checking neighbors and computing cumulative path costs to find the optimal route.
- Suitable for structured, grid-based environments with predefined obstacles.

### Route Planning Using Occupancy Grid Maps with Probabilistic Roadmaps:

- Constructs a Probabilistic Roadmap (PRM) by randomly sampling free-space vertices and connecting them within a predefined distance (d_max).
- Uses a visibility check to ensure connections between nodes are collision-free. (bresenham algorithm for diagonal lines)
- Pathfinding leverages A Search* on the PRM graph, with edge weights defined by Euclidean distances.
