import numpy as np

class BVHNode:
    """Bounding Volume Hierarchy node for spatial acceleration."""
    def __init__(self, triangles=None, bbox_min=None, bbox_max=None):
        self.triangles = triangles or []
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.left = None
        self.right = None
        self.is_leaf = len(self.triangles) <= 4  # Leaf threshold


def build_bvh(nodes: np.ndarray, faces: np.ndarray, triangle_indices: list = None) -> BVHNode:
    """Build a Bounding Volume Hierarchy for spatial acceleration."""
    if triangle_indices is None:
        triangle_indices = list(range(len(faces)))
    
    if len(triangle_indices) == 0:
        return None
    
    # Compute bounding box for all triangles
    all_vertices = []
    for tri_idx in triangle_indices:
        face = faces[tri_idx]
        all_vertices.extend([nodes[face[0], :3], nodes[face[1], :3], nodes[face[2], :3]])
    
    if len(all_vertices) == 0:
        return None
        
    all_vertices = np.array(all_vertices)
    bbox_min = np.min(all_vertices, axis=0)
    bbox_max = np.max(all_vertices, axis=0)
    
    node = BVHNode(triangle_indices, bbox_min, bbox_max)
    
    # If few triangles, make it a leaf
    if len(triangle_indices) <= 4:
        return node
    
    # Find the longest axis to split on
    bbox_size = bbox_max - bbox_min
    split_axis = np.argmax(bbox_size)
    
    # Compute triangle centroids
    centroids = []
    for tri_idx in triangle_indices:
        face = faces[tri_idx]
        centroid = (nodes[face[0], :3] + nodes[face[1], :3] + nodes[face[2], :3]) / 3.0
        centroids.append((centroid[split_axis], tri_idx))
    
    # Sort by centroid position on split axis
    centroids.sort(key=lambda x: x[0])
    
    # Split triangles in half
    mid = len(centroids) // 2
    left_triangles = [tri_idx for _, tri_idx in centroids[:mid]]
    right_triangles = [tri_idx for _, tri_idx in centroids[mid:]]
    
    # Recursively build children
    node.left = build_bvh(nodes, faces, left_triangles)
    node.right = build_bvh(nodes, faces, right_triangles)
    node.is_leaf = False
    
    return node


def ray_bbox_intersect(ray_origin: np.ndarray, ray_direction: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    """Test ray-bounding box intersection using slab method."""
    # Avoid division by zero
    inv_dir = np.where(np.abs(ray_direction) < 1e-8, 1e8, 1.0 / ray_direction)
    
    # Compute intersection distances for each slab
    t1 = (bbox_min - ray_origin) * inv_dir
    t2 = (bbox_max - ray_origin) * inv_dir
    
    # Ensure t1 <= t2 for each axis
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    
    # Find the intersection interval
    tmin_max = np.max(tmin)
    tmax_min = np.min(tmax)
    
    # Ray intersects if tmin_max <= tmax_min and tmax_min >= 0
    return tmin_max <= tmax_min and tmax_min > 1e-8


def ray_triangle_intersect_moller(ray_origin: np.ndarray, ray_direction: np.ndarray, 
                                v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> tuple:
    """Möller-Trumbore ray-triangle intersection test."""
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < 1e-8:
        return False, 0.0  # Ray is parallel to triangle
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return False, 0.0
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    
    if v < 0.0 or u + v > 1.0:
        return False, 0.0
    
    # Compute intersection distance
    t = f * np.dot(edge2, q)
    
    return t > 1e-8, t


def traverse_bvh(bvh_node: BVHNode, ray_origin: np.ndarray, ray_direction: np.ndarray, 
                ray_length: float, nodes: np.ndarray, faces: np.ndarray, 
                exclude_triangle: int = -1) -> bool:
    """Traverse BVH to find ray-triangle intersections."""
    if bvh_node is None:
        return False
    
    # Test ray against bounding box
    if not ray_bbox_intersect(ray_origin, ray_direction, bvh_node.bbox_min, bvh_node.bbox_max):
        return False
    
    if bvh_node.is_leaf:
        # Test ray against all triangles in this leaf
        for tri_idx in bvh_node.triangles:
            if tri_idx == exclude_triangle:
                continue
                
            face = faces[tri_idx]
            v0 = nodes[face[0], :3]
            v1 = nodes[face[1], :3]
            v2 = nodes[face[2], :3]
            
            hit, t = ray_triangle_intersect_moller(ray_origin, ray_direction, v0, v1, v2)
            
            # Only consider intersection if it's CLOSER to camera than target point (t < ray_length)
            # This ensures we only detect triangles that actually occlude the target point
            if hit and t < ray_length - 1e-6:
                return True  # Found occluding intersection
        
        return False
    else:
        # Recursively test children
        if traverse_bvh(bvh_node.left, ray_origin, ray_direction, ray_length, nodes, faces, exclude_triangle):
            return True
        if traverse_bvh(bvh_node.right, ray_origin, ray_direction, ray_length, nodes, faces, exclude_triangle):
            return True
        
        return False
