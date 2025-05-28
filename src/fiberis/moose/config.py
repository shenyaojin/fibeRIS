# fiberis/moose/configs.py
# This file defines configuration classes for hydraulic fractures and stimulated reservoir volumes (SRVs).
# Shenyao Jin, shenyaojin@mines.edu, 2025-05-26
from typing import Optional

class HydraulicFractureConfig:
    """
    Configuration class for defining the properties of a single hydraulic fracture.
    """
    def __init__(self,
                 name: str,
                 length: float,
                 height: float, # In 2D, this often represents the fracture's aperture or thickness for meshing
                 center_x: float,
                 center_y: float,
                 orientation_angle: float = 0.0, # Degrees, relative to X-axis
                 mesh_length_param: Optional[float] = None, # E.g., target element size or refinement level
                 mesh_height_param: Optional[float] = None, # E.g., target element size or refinement level
                 permeability: Optional[float] = None):
        """
        Initializes a HydraulicFractureConfig object.

        Args:
            name (str): Unique name/identifier for the fracture.
            length (float): Length of the fracture (e.g., along the X-axis if orientation_angle is 0).
            height (float): Thickness or aperture of the fracture in the 2D mesh (e.g., its extent in the Y-direction if orientation_angle is 0).
            center_x (float): X-coordinate of the fracture's center point.
            center_y (float): Y-coordinate of the fracture's center point.
            orientation_angle (float, optional): Orientation angle of the fracture in degrees,
                                                 relative to the positive X-axis (counter-clockwise).
                                                 Defaults to 0.0.
                                                 Note: Full geometric implementation of arbitrary angles
                                                 in the ModelBuilder might be deferred.
            mesh_length_param (Optional[float], optional): Mesh refinement parameter along the fracture's length.
                                                            The interpretation (e.g., element count, element size)
                                                            depends on how it's used by the ModelBuilder.
                                                            Defaults to None.
            mesh_height_param (Optional[float], optional): Mesh refinement parameter along the fracture's height/aperture.
                                                             Defaults to None.
            permeability (Optional[float], optional): Permeability of the fracture zone. Defaults to None.
        """
        self.name: str = name
        self.length: float = length
        self.height: float = height
        self.center_x: float = center_x
        self.center_y: float = center_y
        self.orientation_angle: float = orientation_angle
        self.mesh_length_param: Optional[float] = mesh_length_param
        self.mesh_height_param: Optional[float] = mesh_height_param
        self.permeability: Optional[float] = permeability

        if self.orientation_angle != 0.0:
            print(f"Info: HydraulicFractureConfig '{self.name}' has a non-zero orientation_angle ({self.orientation_angle}°). "
                  "Full geometric support for arbitrary angles in the mesh generation process might be limited "
                  "or implemented in later stages. Current implementation might assume axis-aligned features.")

class SRVConfig:
    """
    Configuration class for defining the properties of a Stimulated Reservoir Volume (SRV) zone.
    """
    def __init__(self,
                 name: str,
                 length: float, # Dimension along X-axis by default
                 width: float,  # Dimension along Y-axis by default
                 center_x: float,
                 center_y: float,
                 mesh_length_param: Optional[float] = None, # E.g., target element size or refinement level along length
                 mesh_width_param: Optional[float] = None,  # E.g., target element size or refinement level along width
                 permeability: Optional[float] = None):
        """
        Initializes an SRVConfig object.

        Args:
            name (str): Unique name/identifier for the SRV zone.
            length (float): Length of the SRV zone (e.g., along X-axis).
            width (float): Width of the SRV zone (e.g., along Y-axis).
            center_x (float): X-coordinate of the SRV zone's center point.
            center_y (float): Y-coordinate of the SRV zone's center point.
            mesh_length_param (Optional[float], optional): Mesh size or refinement parameter along the SRV zone's length.
                                                            Defaults to None.
            mesh_width_param (Optional[float], optional): Mesh size or refinement parameter along the SRV zone's width.
                                                           Defaults to None.
            permeability (Optional[float], optional): Permeability of the SRV zone. Defaults to None.
        """
        self.name: str = name
        self.length: float = length
        self.width: float = width
        self.center_x: float = center_x
        self.center_y: float = center_y
        self.mesh_length_param: Optional[float] = mesh_length_param
        self.mesh_width_param: Optional[float] = mesh_width_param
        self.permeability: Optional[float] = permeability

if __name__ == '__main__':
    # Example Usage:
    fracture1_config = HydraulicFractureConfig(
        name="MainFracture",
        length=200.0,
        height=0.1,
        center_x=500.0,
        center_y=250.0,
        orientation_angle=0.0, # Set to 0 for now as per discussion
        mesh_length_param=5.0,
        permeability=1.0e-12
    )

    srv1_config = SRVConfig(
        name="PrimarySRV",
        length=400.0,
        width=100.0,
        center_x=500.0,
        center_y=250.0,
        mesh_length_param=10.0,
        permeability=1.0e-15
    )

    print(f"Fracture Config ('{fracture1_config.name}'):")
    print(f"  Length: {fracture1_config.length}, Height/Aperture: {fracture1_config.height}")
    print(f"  Center: ({fracture1_config.center_x}, {fracture1_config.center_y})")
    print(f"  Orientation: {fracture1_config.orientation_angle}°")
    print(f"  Permeability: {fracture1_config.permeability}")

    print(f"\nSRV Config ('{srv1_config.name}'):")
    print(f"  Length: {srv1_config.length}, Width: {srv1_config.width}")
    print(f"  Center: ({srv1_config.center_x}, {srv1_config.center_y})")
    print(f"  Permeability: {srv1_config.permeability}")
