# fiberis/moose/configs.py
# This file defines configuration classes for hydraulic fractures and stimulated reservoir volumes (SRVs).
# Shenyao Jin, shenyaojin@mines.edu, 2025-05-26
from typing import List, Dict, Any, Optional

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
                 height: float,  # Dimension along Y-axis by default
                 center_x: float,
                 center_y: float,
                 mesh_length_param: Optional[float] = None, # E.g., target element size or refinement level along length
                 mesh_height_param: Optional[float] = None,  # E.g., target element size or refinement level along height
                 permeability: Optional[float] = None):
        """
        Initializes an SRVConfig object.

        Args:
            name (str): Unique name/identifier for the SRV zone.
            length (float): Length of the SRV zone (e.g., along X-axis).
            height (float): height of the SRV zone (e.g., along Y-axis).
            center_x (float): X-coordinate of the SRV zone's center point.
            center_y (float): Y-coordinate of the SRV zone's center point.
            mesh_length_param (Optional[float], optional): Mesh size or refinement parameter along the SRV zone's length.
                                                            Defaults to None.
            mesh_height_param (Optional[float], optional): Mesh size or refinement parameter along the SRV zone's height.
                                                           Defaults to None.
            permeability (Optional[float], optional): Permeability of the SRV zone. Defaults to None.
        """
        self.name: str = name
        self.length: float = length
        self.height: float = height
        self.center_x: float = center_x
        self.center_y: float = center_y
        self.mesh_length_param: Optional[float] = mesh_length_param
        self.mesh_height_param: Optional[float] = mesh_height_param
        self.permeability: Optional[float] = permeability

class IndicatorConfig:
    """
    Configuration for a single Indicator within the MOOSE [Adaptivity][Indicators] block.
    An indicator computes an error estimate or other metric for each element.
    """
    def __init__(self,
                 name: str,
                 type: str,
                 params: Dict[str, Any]):
        """
        Initializes an IndicatorConfig object.

        Args:
            name (str): The user-chosen name for this indicator (e.g., "error").
                        This name is used locally within the [Adaptivity] block,
                        for example, to be referenced by a Marker.
            type (str): The MOOSE type for the indicator (e.g., "GradientJumpIndicator",
                        "ErrorResidualIndicator").
            params (Dict[str, Any]): A dictionary of parameters for this indicator.
                                     Example: {"variable": "convected", "outputs": "none"}
        """
        self.name: str = name
        self.type: str = type
        self.params: Dict[str, Any] = params

class MarkerConfig:
    """
    Configuration for a single Marker within the MOOSE [Adaptivity][Markers] block.
    A marker determines which elements to refine or coarsen based on values from an indicator.
    """
    def __init__(self,
                 name: str,
                 type: str,
                 params: Dict[str, Any]):
        """
        Initializes a MarkerConfig object.

        Args:
            name (str): The user-chosen name for this marker (e.g., "errorfrac").
                        This name is used by the main [Adaptivity] block's 'marker' parameter.
            type (str): The MOOSE type for the marker (e.g., "ErrorFractionMarker",
                        "ValueThresholdMarker").
            params (Dict[str, Any]): A dictionary of parameters for this marker.
                                     Example: {"indicator": "error", "refine": 0.5, "coarsen": 0}
                                     The "indicator" key should refer to the 'name' of an IndicatorConfig.
        """
        self.name: str = name
        self.type: str = type
        self.params: Dict[str, Any] = params

class AdaptivityConfig:
    """
    Configuration for the MOOSE [Adaptivity] top-level block.
    Manages indicators, markers, and overall adaptivity settings.
    """
    def __init__(self,
                 marker_to_use: str,
                 steps: int,
                 indicators: Optional[List[IndicatorConfig]] = None,
                 markers: Optional[List[MarkerConfig]] = None):
        """
        Initializes an AdaptivityConfig object.

        Args:
            marker_to_use (str): The name of the MarkerConfig instance (defined in the 'markers' list)
                                 that the [Adaptivity] block should use. This corresponds to the
                                 'marker = ...' parameter in the [Adaptivity] block.
            steps (int): The number of adaptivity steps to perform during the simulation.
                         Corresponds to the 'steps = ...' parameter.
            indicators (Optional[List[IndicatorConfig]], optional): A list of IndicatorConfig objects.
                                                                    Defaults to None, meaning no indicators
                                                                    will be explicitly defined by this object initially.
            markers (Optional[List[MarkerConfig]], optional): A list of MarkerConfig objects.
                                                              Defaults to None.
        """
        self.marker_to_use: str = marker_to_use
        self.steps: int = steps
        self.indicators: List[IndicatorConfig] = indicators if indicators is not None else []
        self.markers: List[MarkerConfig] = markers if markers is not None else []

    def add_indicator(self, indicator_config: IndicatorConfig):
        """Adds an indicator configuration to this adaptivity setup."""
        if not isinstance(indicator_config, IndicatorConfig):
            raise TypeError("indicator_config must be an instance of IndicatorConfig.")
        self.indicators.append(indicator_config)

    def add_marker(self, marker_config: MarkerConfig):
        """Adds a marker configuration to this adaptivity setup."""
        if not isinstance(marker_config, MarkerConfig):
            raise TypeError("marker_config must be an instance of MarkerConfig.")
        self.markers.append(marker_config)

if __name__ == '__main__':
    # Example usage of the new AMA configuration classes:

    # 1. Define an indicator
    error_indicator = IndicatorConfig(
        name="error_on_pp",
        type="GradientJumpIndicator",
        params={"variable": "pp", "outputs": "none"} # Assuming 'pp' is a defined variable
    )

    # 2. Define another indicator (optional)
    another_indicator = IndicatorConfig(
        name="solution_indicator",
        type="ValueJumpIndicator", # Example type
        params={"variable": "disp_x"}
    )

    # 3. Define a marker that uses the first indicator
    error_fraction_marker = MarkerConfig(
        name="errorfrac_pp_marker",
        type="ErrorFractionMarker",
        params={"indicator": "error_on_pp", "refine": 0.6, "coarsen": 0.1, "outputs": "none"}
    )

    # 4. Define another marker (optional)
    threshold_marker = MarkerConfig(
        name="disp_x_threshold",
        type="ValueThresholdMarker", # Example type
        params={"indicator": "solution_indicator", "min_value": 0.01, "max_value": 0.5, "refine": True}
    )


    # 5. Define the overall adaptivity configuration
    # Option A: Initialize with empty lists and add later
    ama_setup_option_a = AdaptivityConfig(
        marker_to_use="errorfrac_pp_marker", # This name must match one of the MarkerConfig names
        steps=3
    )
    ama_setup_option_a.add_indicator(error_indicator)
    ama_setup_option_a.add_marker(error_fraction_marker)
    # Optionally add more
    ama_setup_option_a.add_indicator(another_indicator)
    ama_setup_option_a.add_marker(threshold_marker)


    # Option B: Initialize directly with lists of configs
    ama_setup_option_b = AdaptivityConfig(
        marker_to_use="errorfrac_pp_marker",
        steps=2,
        indicators=[
            IndicatorConfig(name="pressure_grad", type="GradientJumpIndicator", params={"variable": "pp"}),
            IndicatorConfig(name="stress_error", type="SomeOtherErrorIndicator", params={"variable": "von_mises_stress"})
        ],
        markers=[
            MarkerConfig(name="errorfrac_pp_marker", type="ErrorFractionMarker", params={"indicator": "pressure_grad", "refine": 0.5}),
            MarkerConfig(name="stress_based_marker", type="ErrorFractionMarker", params={"indicator": "stress_error", "refine": 0.4})
        ]
    )


    print("Adaptivity Configuration Example (Option A):")
    print(f"  Main Adaptivity Settings: marker='{ama_setup_option_a.marker_to_use}', steps={ama_setup_option_a.steps}")
    for ind in ama_setup_option_a.indicators:
        print(f"    Indicator: name='{ind.name}', type='{ind.type}', params={ind.params}")
    for marker in ama_setup_option_a.markers:
        print(f"    Marker: name='{marker.name}', type='{marker.type}', params={marker.params}")

    print("\nAdaptivity Configuration Example (Option B):")
    print(f"  Main Adaptivity Settings: marker='{ama_setup_option_b.marker_to_use}', steps={ama_setup_option_b.steps}")
    for ind in ama_setup_option_b.indicators:
        print(f"    Indicator: name='{ind.name}', type='{ind.type}', params={ind.params}")
    for marker in ama_setup_option_b.markers:
        print(f"    Marker: name='{marker.name}', type='{marker.type}', params={marker.params}")
