# fiberis/moose/configs.py
# This file defines configuration classes for hydraulic fractures and stimulated reservoir volumes (SRVs).
# Shenyao Jin, shenyaojin@mines.edu, 2025-05-26
from typing import List, Dict, Any, Optional, Union, Tuple

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


class PostprocessorConfigBase:
    """
    Base configuration class for a MOOSE Postprocessor.
    """

    def __init__(self,
                 name: str,
                 pp_type: str,  # MOOSE type of the postprocessor
                 execute_on: Optional[Union[str, List[str]]] = None,
                 variable: Optional[str] = None,  # For postprocessors operating on a single variable
                 variables: Optional[List[str]] = None,  # For postprocessors operating on multiple variables
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the base configuration for a postprocessor.

        Args:
            name (str): The user-chosen name for this postprocessor.
            pp_type (str): The MOOSE type string for the postprocessor.
            execute_on (Optional[Union[str, List[str]]], optional): When to execute.
                Defaults to None (MOOSE default, often 'timestep_end').
            variable (Optional[str], optional): Single variable to operate on. Defaults to None.
            variables (Optional[List[str]], optional): List of variables to operate on. Defaults to None.
            other_params (Optional[Dict[str, Any]], optional): Dictionary for any other specific parameters.
                                                               Defaults to None.
        """
        self.name: str = name
        self.pp_type: str = pp_type  # This will be set by derived classes typically
        self.execute_on: Optional[Union[str, List[str]]] = execute_on
        self.variable: Optional[str] = variable
        self.variables: Optional[List[str]] = variables
        self.other_params: Dict[str, Any] = other_params if other_params is not None else {}


class PointValueSamplerConfig(PostprocessorConfigBase):
    """
    Configuration for a 'PointValue' or similar point-based scalar MOOSE Postprocessor.
    """

    def __init__(self,
                 name: str,
                 variable: str,  # PointValue typically operates on a single variable
                 point: Union[Tuple[float, ...], str],  # (x,y) or (x,y,z) or "x y z"
                 execute_on: Optional[Union[str, List[str]]] = None,
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Initializes a PointValueSamplerConfig object.

        Args:
            name (str): Name for this postprocessor.
            variable (str): The variable to sample.
            point (Union[Tuple[float, ...], str]): Coordinates of the point to sample.
            execute_on (Optional[Union[str, List[str]]], optional): When to execute.
            other_params (Optional[Dict[str, Any]], optional): Additional parameters.
        """
        super().__init__(name=name,
                         pp_type="PointValue",  # MOOSE type for this specific config
                         execute_on=execute_on,
                         variable=variable,
                         other_params=other_params)

        if not isinstance(point, str):
            self.point: str = ' '.join(map(str, point))
        else:
            self.point: str = point


class LineValueSamplerConfig(PostprocessorConfigBase):
    """
    Configuration for a 'LineValueSampler' MOOSE Postprocessor.
    This can output scalar statistics (min, max, avg) or a vector of values along the line.
    """

    def __init__(self,
                 name: str,
                 variable: str,
                 start_point: Union[Tuple[float, ...], str],
                 end_point: Union[Tuple[float, ...], str],
                 num_points: int = 100,  # Default number of sample points on the line
                 output_vector: bool = False,  # If true, configures for vector output (e.g., to CSV)
                 execute_on: Optional[Union[str, List[str]]] = None,
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Initializes a LineValueSamplerConfig object.

        Args:
            name (str): Name for this postprocessor.
            variable (str): The variable to sample along the line.
            start_point (Union[Tuple[float, ...], str]): Start coordinates of the line.
            end_point (Union[Tuple[float, ...], str]): End coordinates of the line.
            num_points (int, optional): Number of points to sample along the line. Defaults to 100.
            output_vector (bool, optional): If True, sets 'output_vector_postprocessor = true'
                                            in other_params, typically for CSV output of all points.
                                            Otherwise, it might output scalar stats like min/max/avg.
                                            Defaults to False.
            execute_on (Optional[Union[str, List[str]]], optional): When to execute.
            other_params (Optional[Dict[str, Any]], optional): Additional parameters.
        """
        # Initialize other_params first if it's None
        current_other_params = other_params if other_params is not None else {}

        # Add LineValueSampler specific parameters to other_params
        if not isinstance(start_point, str):
            current_other_params["start_point"] = ' '.join(map(str, start_point))
        else:
            current_other_params["start_point"] = start_point

        if not isinstance(end_point, str):
            current_other_params["end_point"] = ' '.join(map(str, end_point))
        else:
            current_other_params["end_point"] = end_point

        current_other_params["num_points"] = num_points

        if output_vector:
            current_other_params["output_vector_postprocessor"] = True  # MOOSE boolean

        super().__init__(name=name,
                         pp_type="LineValueSampler",  # MOOSE type
                         execute_on=execute_on,
                         variable=variable,
                         other_params=current_other_params)

        # Make these accessible as direct attributes as well for clarity, though they are in other_params
        self.start_point_str: str = current_other_params["start_point"]
        self.end_point_str: str = current_other_params["end_point"]
        self.num_sample_points: int = num_points


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
