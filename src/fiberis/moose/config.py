# fiberis/moose/configs.py
# This file defines configuration classes for hydraulic fractures and stimulated reservoir volumes (SRVs).
# Shenyao Jin, shenyaojin@mines.edu, 2025-05-26
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np

# Import the Data1D base class to be used for subclassing
from fiberis.analyzer.Data1D.core1D import Data1D

# +++ Material Properties +++
class ZoneMaterialProperties:
    """
    A data container for the physical properties of a single mesh zone/block.
    This object will be part of a MatrixConfig, SRVConfig, or HydraulicFractureConfig.
    """
    def __init__(self,
                 porosity: float,
                 permeability: Union[float, str], # Can be a scalar, tensor string, or filepath string
                 youngs_modulus: Optional[float] = None, # For solid mechanics
                 poissons_ratio: Optional[float] = None): # For solid mechanics
        """
        Initializes material properties for a specific zone.

        Args:
            porosity (float): The porosity of the material in this zone.
            permeability (Union[float, str]): The permeability. Can be:
                - A float for constant, isotropic permeability.
                - A string for a constant, anisotropic tensor (e.g., '1e-15 0 0 ...').
                - A string representing a filepath to an .npz file for time-dependent permeability.
            youngs_modulus (Optional[float], optional): Young's modulus for elasticity. Defaults to None.
            poissons_ratio (Optional[float], optional): Poisson's ratio for elasticity. Defaults to None.
        """
        self.porosity: float = porosity
        self.permeability: Union[float, str] = permeability
        self.youngs_modulus: Optional[float] = youngs_modulus
        self.poissons_ratio: Optional[float] = poissons_ratio

# +++ Matrix Configuration +++
class MatrixConfig:
    """
    Configuration for the main reservoir matrix, including its name and material properties.
    """
    def __init__(self,
                 name: str,
                 materials: ZoneMaterialProperties):
        """
        Initializes the MatrixConfig.

        Args:
            name (str): The name for the matrix block (e.g., "matrix").
            materials (ZoneMaterialProperties): An object containing the material properties for the matrix.
        """
        self.name: str = name
        self.materials: ZoneMaterialProperties = materials


# +++ HydraulicFractureConfig +++
class HydraulicFractureConfig:
    """
    UPDATED: Configuration for a hydraulic fracture, now including material properties.
    """
    def __init__(self,
                 name: str,
                 length: float,
                 height: float, # In 2D, this often represents the fracture's aperture or thickness for meshing
                 center_x: float,
                 center_y: float,
                 materials: ZoneMaterialProperties, # <-- Replaces individual material properties
                 orientation_angle: float = 0.0, # Degrees, relative to X-axis
                 mesh_length_param: Optional[float] = None, # E.g., target element size or refinement level
                 mesh_height_param: Optional[float] = None): # E.g., target element size or refinement level
        """
        Initializes a HydraulicFractureConfig object.

        Args:
            name (str): Unique name/identifier for the fracture.
            length (float): Length of the fracture (e.g., along the X-axis if orientation_angle is 0).
            height (float): Thickness or aperture of the fracture in the 2D mesh.
            center_x (float): X-coordinate of the fracture's center point.
            center_y (float): Y-coordinate of the fracture's center point.
            materials (ZoneMaterialProperties): An object containing material properties for the fracture.
            orientation_angle (float, optional): Orientation angle of the fracture in degrees. Defaults to 0.0.
            mesh_length_param (Optional[float], optional): Mesh refinement parameter along the fracture's length.
            mesh_height_param (Optional[float], optional): Mesh refinement parameter along the fracture's height/aperture.
        """
        self.name: str = name
        self.length: float = length
        self.height: float = height
        self.center_x: float = center_x
        self.center_y: float = center_y
        self.materials: ZoneMaterialProperties = materials # <-- New attribute
        self.orientation_angle: float = orientation_angle
        self.mesh_length_param: Optional[float] = mesh_length_param
        self.mesh_height_param: Optional[float] = mesh_height_param

        if self.orientation_angle != 0.0:
            print(f"Info: HydraulicFractureConfig '{self.name}' has a non-zero orientation_angle ({self.orientation_angle}°). "
                  "Full geometric support for arbitrary angles in the mesh generation process might be limited.")


# +++ SRVConfig +++
class SRVConfig:
    """
    UPDATED: Configuration for an SRV zone, now including material properties.
    """
    def __init__(self,
                 name: str,
                 length: float, # Dimension along X-axis by default
                 height: float, # Dimension along Y-axis by default
                 center_x: float,
                 center_y: float,
                 materials: ZoneMaterialProperties, # <-- Replaces individual material properties
                 mesh_length_param: Optional[float] = None, # E.g., target element size or refinement level along length
                 mesh_height_param: Optional[float] = None): # E.g., target element size or refinement level along height
        """
        Initializes an SRVConfig object.

        Args:
            name (str): Unique name/identifier for the SRV zone.
            length (float): Length of the SRV zone (e.g., along X-axis).
            height (float): Height of the SRV zone (e.g., along Y-axis).
            center_x (float): X-coordinate of the SRV zone's center point.
            center_y (float): Y-coordinate of the SRV zone's center point.
            materials (ZoneMaterialProperties): An object containing material properties for the SRV.
            mesh_length_param (Optional[float], optional): Mesh size or refinement parameter along the SRV zone's length.
            mesh_height_param (Optional[float], optional): Mesh size or refinement parameter along the SRV zone's height.
        """
        self.name: str = name
        self.length: float = length
        self.height: float = height
        self.center_x: float = center_x
        self.center_y: float = center_y
        self.materials: ZoneMaterialProperties = materials # <-- New attribute
        self.mesh_length_param: Optional[float] = mesh_length_param
        self.mesh_height_param: Optional[float] = mesh_height_param

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
                 pp_type: str, # MOOSE type of the postprocessor
                 execute_on: Optional[Union[str, List[str]]] = ['initial', 'timestep_end', 'final'],
                 variable: Optional[str] = None, # For postprocessors operating on a single variable
                 variables: Optional[List[str]] = None, # For postprocessors operating on multiple variables
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the base configuration for a postprocessor.
        """
        self.name: str = name
        self.pp_type: str = pp_type
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
                 variable: str, # PointValue typically operates on a single variable
                 point: Union[Tuple[float, ...], str], # (x,y) or (x,y,z) or "x y z"
                 execute_on: Optional[Union[str, List[str]]] = None,
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Initializes a PointValueSamplerConfig object.
        """
        super().__init__(name=name,
                         pp_type="PointValue", # MOOSE type for this specific config
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
                 num_points: int = 100, # Default number of sample points on the line
                 output_vector: bool = False, # If true, configures for vector output (e.g., to CSV)
                 execute_on: Optional[Union[str, List[str]]] = None,
                 other_params: Optional[Dict[str, Any]] = None):
        """
        Initializes a LineValueSamplerConfig object.
        """
        current_other_params = other_params if other_params is not None else {}
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
            current_other_params["output_vector_postprocessor"] = True # MOOSE boolean

        super().__init__(name=name,
                         pp_type="LineValueSampler", # MOOSE type
                         execute_on=execute_on,
                         variable=variable,
                         other_params=current_other_params)
        self.start_point_str: str = current_other_params["start_point"]
        self.end_point_str: str = current_other_params["end_point"]
        self.num_sample_points: int = num_points


class PostprocessorConfig(PostprocessorConfigBase):
    """
    A generic configuration class for any MOOSE Postprocessor.
    This provides flexibility for postprocessors not covered by specific configs.
    """
    def __init__(self,
                 name: str,
                 pp_type: str,
                 params: Optional[Dict[str, Any]] = None):
        """
        Initializes a generic PostprocessorConfig.

        Args:
            name (str): The name of the postprocessor block.
            pp_type (str): The MOOSE type of the postprocessor (e.g., 'ElementAverageValue').
            params (Optional[Dict[str, Any]], optional): A dictionary of parameters for the postprocessor.
        """
        # Extract standard parameters from the params dict to pass to the base class
        execute_on = params.pop('execute_on', None)
        variable = params.pop('variable', None)
        variables = params.pop('variables', None)

        super().__init__(name=name,
                         pp_type=pp_type,
                         execute_on=execute_on,
                         variable=variable,
                         variables=variables,
                         other_params=params)


class SimpleFluidPropertiesConfig:
    """
    Configuration for a set of constant fluid properties, designed to be used with
    the 'SimpleFluidProperties' material in a MOOSE [FluidProperties] block.
    Default values are provided for a water-like fluid.
    """
    def __init__(self,
                 name: str,
                 bulk_modulus: float = 2.2E9,
                 viscosity: float = 1.0E-3,
                 density0: float = 1000.0,
                 thermal_expansion: Optional[float] = 0.0002,
                 cp: Optional[float] = 4194.0, # Specific heat at constant pressure
                 cv: Optional[float] = 4186.0, # Specific heat at constant volume
                 porepressure_coefficient: Optional[float] = 1.0):
        """
        Initializes a SimpleFluidPropertiesConfig object.
        """
        self.name: str = name
        self.bulk_modulus: float = bulk_modulus
        self.viscosity: float = viscosity
        self.density0: float = density0
        self.thermal_expansion: Optional[float] = thermal_expansion
        self.cp: Optional[float] = cp
        self.cv: Optional[float] = cv
        self.porepressure_coefficient: Optional[float] = porepressure_coefficient

# Adaptive time-stepping configuration
class TimeStepperFunctionConfig(Data1D):
    """
    Refactored configuration class for a timestep limiting function.

    This class now inherits from Data1D, treating the function's time points
    as 'taxis' and the corresponding dt values as 'data'. This improves code
    consistency and reusability.
    """
    def __init__(self,
                 name: str,
                 x_values: Union[List[float], np.ndarray],
                 y_values: Union[List[float], np.ndarray]):
        """
        Initializes the TimeStepperFunctionConfig.

        Args:
            name (str): The name of the function in the MOOSE input file (e.g., 'constant_step_1').
            x_values (Union[List[float], np.ndarray]): A list of time points. This corresponds to the 'x' axis.
            y_values (Union[List[float], np.ndarray]): A list of corresponding timestep (dt) values. This corresponds to the 'y' axis.
        """
        # Call the parent's __init__ method, mapping x/y to taxis/data
        super().__init__(name=name, taxis=np.array(x_values), data=np.array(y_values))

    @classmethod
    def load_timestep_from_data1d(cls, name: str, data1d_obj: Data1D) -> 'TimeStepperFunctionConfig':
        """
        Creates a TimeStepperFunctionConfig from a Data1D object.

        The x_values are taken directly from the Data1D object's taxis.
        The y_values (timesteps) are calculated as half the difference between consecutive taxis points.
        The last y_value is duplicated to match the length of the x_values.

        Args:
            name (str): The name for the new TimeStepperFunctionConfig.
            data1d_obj (Data1D): The Data1D object to process.

        Returns:
            TimeStepperFunctionConfig: A new instance configured with the derived timesteps.
        
        Raises:
            ValueError: If the Data1D object has fewer than 2 time points.
        """
        if data1d_obj.taxis is None or len(data1d_obj.taxis) < 2:
            raise ValueError("Cannot generate timesteps from Data1D object with fewer than 2 time points.")

        x_values = data1d_obj.taxis
        
        # Calculate y_values as half the difference of taxis
        dt_values = np.diff(x_values) / 2.0
        
        # Append the last element to make the arrays the same size
        y_values = np.append(dt_values, dt_values[-1])
        
        return cls(name=name, x_values=x_values, y_values=y_values)


@dataclass
class AdaptiveTimeStepperConfig:
    """
    A container class for configuring the IterationAdaptiveDT TimeStepper.

    It holds a list of all the schedule functions that the time stepper will use
    to limit the timestep throughout the simulation.
    """
    # This list will contain all the "schedule" functions,
    # which are now instances of the new TimeStepperFunctionConfig class.
    functions: List[TimeStepperFunctionConfig]


class TimeSequenceStepper:
    """
    Configuration for the TimeSequenceStepper in MOOSE.
    This time stepper derives from TimeSequenceStepperBase and provides the sequence of time values from a user-specified list, given by "time_sequence".

    See Failed solves for information on the behavior of this time stepper for failed time steps.

    Example input syntax
    In this example, the numerical problem is solved at four specified points in time using a TimeSequenceStepper.

    [Executioner]
      type = Transient
      end_time = 4.0
      [./TimeStepper]
        type = TimeSequenceStepper
        time_sequence = '0   0.85 1.3 2 4'
      [../]
    []
    """
    def __init__(self, time_sequence: Optional[Union[List[float], np.ndarray]] = None):
        """
        Initializes the TimeSequenceStepper.

        Args:
            time_sequence (Optional[Union[List[float], np.ndarray]]): A list or array of time points for the simulation. Can be provided later.
        """
        if time_sequence is not None:
            if not isinstance(time_sequence, (list, np.ndarray)):
                raise TypeError("time_sequence must be a list or numpy array.")
            # MOOSE expects a space-separated string
            self.time_sequence: str = ' '.join(map(str, time_sequence))
        else:
            self.time_sequence: str = '' # Initialize as empty string if not provided

    def from_data1d(self, data1d_obj: Data1D) -> None:
        """
        Loads the time sequence from a Data1D object's time axis, modifying the instance in place.

        Args:
            data1d_obj (Data1D): The Data1D object containing the time axis (taxis).
        """
        if data1d_obj.taxis is None or data1d_obj.taxis.size == 0:
            raise ValueError("Data1D object must have a non-empty taxis to create a TimeSequenceStepper.")
        self.time_sequence = ' '.join(map(str, data1d_obj.taxis))