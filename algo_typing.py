from typing import Tuple, List

CC = Tuple[List[int], List[int]] # Connected Component (CC) of labeled lesions
IndexExpression3D = Tuple[slice, slice, slice] # Index Expression for 3D numpy array slicing (x, y, z)
VoxelSpacing = Tuple[float, float, float] # Voxel spacing in x, y, z direction (mm)
