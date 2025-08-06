# The input file for the MOOSE framework. From Victor Fakeye, RCP PhD candidate
# EGS project, Fervo Energy.

[Mesh]
  [gen1]
    type = GeneratedMeshGenerator
    dim = 2
    bias_y = 1.3
    ny = 40
    nx = 200
    xmax = 1000
    ymin = 0
    ymax = 500
  []

  [gen2]
    type = GeneratedMeshGenerator
    dim = 2
    bias_y = 1.3
    ny = 40
    nx = 200
    xmin = 0
    xmax = 1000
    ymin = 0
    ymax = 500
  []

  [gen]
    type = TransformGenerator
    input = gen2
    transform = ROTATE
    vector_value = '0 180 0'
  []

  [smg]
    type = StitchedMeshGenerator
    inputs = 'gen1 gen'
    clear_stitched_boundary_ids = true
    stitch_boundaries_pairs = 'bottom bottom'
  []

  [SRV]
    type = SubdomainBoundingBoxGenerator
    input = smg
    bottom_left = '350 -8 0.00000001'
    top_right = '650 8 0'
    block_id = 1
  []

  [SRV2]
    type = SubdomainBoundingBoxGenerator
    input = SRV
    bottom_left = '450 -12 0.00000001'
    top_right = '550 12 0'
    block_id = 3
  []

  [SRV3]
    type = SubdomainBoundingBoxGenerator
    input = SRV2
    bottom_left = '400 -10 0.00000001'
    top_right = '600 10 0'
    block_id = 4
  []

  [fracture]
    type = SubdomainBoundingBoxGenerator
    input = SRV3
    bottom_left = '375 -0.01 0.00000001'
    top_right = '625 0.01 0'
    block_id = 2
  []

  [refine]
    type = RefineBlockGenerator
    input = fracture
    refinement = '1 1'
    block = '3 1'
  []

  [refine2]
    type = RefineBlockGenerator
    input = refine
    refinement = '1 1'
    block = '4 2'
  []


  [injection_point]
    input = refine2
    type = ExtraNodesetGenerator
    new_boundary = injection_well
    coord = '500 0'
  []

  [production_point]
    input = injection_point
    type = ExtraNodesetGenerator
    new_boundary = production_well
    coord = '610 0'
  []

  [rename]
    type = RenameBlockGenerator
    old_block = '0 1 2 3 4'
    new_block = 'matrix srv fracture srv2 srv3'
    input = production_point
  []
[]

[GlobalParams]
  PorousFlowDictator = dictator
  displacements = 'disp_x disp_y'
  #gravity = '0 0 0'
[]

[Variables]
  [pp]
  initial_condition = 26.4E6
  []
  [disp_x]
  scaling = 1E-10
  []
  [disp_y]
  scaling = 1E-10
  []
[]

[Kernels]
  [dot]
    type = TimeDerivative
    variable = pp
  []

  [srv_diffusion]
    type = FunctionDiffusion
    variable = pp
    function = srv_diff
    block = 'srv srv2 srv3'
  []

  [fracture_diffusion]
    type = FunctionDiffusion
    variable = pp
    function = frac_diff
    block = fracture
  []

  #[fracture_diffusion]
  #  type = AnisotropicDiffusion
  #  block = fracture
  #  tensor_coeff = '35 0 0  0 35 0  0 0 35' #Initial: 11.8
  #  variable = pp
  #[]

  #[srv_diffusion]
  #  type = AnisotropicDiffusion
  #  block = 'srv srv2'
  #  tensor_coeff = '0.001 0 0  0 0.001 0  0 0 0.001' #10ud =0.000217
  #  variable = pp
  #[]

  [matrix_diffusion]
    type = AnisotropicDiffusion
    block = matrix
    tensor_coeff = '0.00005894 0 0  0 0.00005894 0  0 0 0.00005894'
    variable = pp
  []

  [flux]
    type = PorousFlowFullySaturatedDarcyBase
    variable = pp
    gravity = '0 0 0'
  []

  [grad_stress_x]
    type = StressDivergenceTensors
    variable = disp_x
    component = 0
  []

  [grad_stress_y]
    type = StressDivergenceTensors
    variable = disp_y
    component = 1
  []

  [poro_x]
    type = PorousFlowEffectiveStressCoupling
    biot_coefficient = 0.7
    variable = disp_x
    component = 0
  []

  [poro_y]
    type = PorousFlowEffectiveStressCoupling
    biot_coefficient = 0.7
    variable = disp_y
    component = 1
  []

  [vol_strain_rate_water]
    type = PorousFlowMassVolumetricExpansion
    fluid_component = 0
    variable = pp
  []
[]

[BCs]
  [injection_pressure]
    type = FunctionDirichletBC
    variable = pp
    boundary = injection_well
    function = pres_func
  []

  [confinex]
    type = DirichletBC
    variable = disp_x
    value = 0
    boundary = 'left right'
  []

  [confiney]
    type = DirichletBC
    variable = disp_y
    value = 0
    boundary = 'top'
  []
[]

[Functions]
 [pres_func]
    type = ParsedFunction
    expression = 'if(t <= 26000, 26.5E6, if(t <= 353000, 27.5E6 + (31E6 - 27.5E6) * (1 - exp(-6e-6 * t)), if(t <= 356200, 32.3E6, if(t <= 359200, 34.2E6, if(t <= 363800, 36.2E6, if(t <= 379451, 36.2E6 + (39.5E6 - 36.2E6) * (1 - exp(-2.5e-4 * (t - 363800))), 27.5E6 + (30.5E6 - 27.5E6) * exp(-(t - 378351) / 12700)))))))'
 []

 [frac_diff]
    type = ParsedFunction
    expression = 'if(t <= 379451, 33.5, 5)'
 []

 #[frac_diff]
 #   type = ParsedFunction
 #   expression = '60 * exp(1e-9 * beta)'
 #   symbol_names = 'beta'
 #   symbol_values = pp_inj
 #[]

 #[srv_diff]
 #   type = ParsedFunction
 #   expression = 'if(t <= 368000, 0.000383 * exp(10e-9 * alpha), if(t < 379451, 0.000383 * exp(10e-9 * alpha), if(t = 379451, 0.000383 * exp(10e-9 * alpha), 0.000383 * exp(10e-9 * pres_func_max))))'
 #   symbol_names = 'alpha pres_func_max'
 #   symbol_values = 'pres_func 39.5E6'
 #[]

 [srv_diff]
    type = ParsedFunction
    expression = 'if(t <= 379451, 0.00082, 0.0023)'
 []

 #[srv_diff]
 #   type = ParsedFunction
 #   expression = 'if(t <= 379451, 0.0067 * exp(3e-9 * alpha), 0.008 * exp(15e-9 * pres_func_max))'
 #   symbol_names = 'alpha pres_func_max'
 #   symbol_values = 'pres_func 39.5E6'
 #[]

 #[srv_diff]
 #   type = ParsedFunction
 #   expression = 'if(t <= 353000, base_diff * exp(comp1 * alpha), if(t <= 356200, base_diff * exp(comp2 * alpha), if(t <= 359200, base_diff * exp(comp3 * alpha), if(t <= 363800, base_diff * exp(comp4 * alpha), 0.002 * exp(comp5 * pres_func_max)))))'
 #   symbol_names = 'alpha base_diff comp1 comp2 comp3 comp4 comp5 pres_func_max'
 #   symbol_values = 'pres_func 0.00065 3e-9 3e-9 3e-9 3e-9 3e-9 39.5e6'
 #[]

 #[diff_func2]
 #   type = ParsedFunction
 #   expression = 'if(t <= 379451, 0.00217 * exp(5e-9 * alpha), 0.01)'
 #   symbol_names = 'alpha'
 #   symbol_values = pp_mon7
 #[]
[]

[UserObjects]
  [dictator]
    type = PorousFlowDictator
    porous_flow_vars = 'pp'
    number_fluid_phases = 1
    number_fluid_components = 1
  []
[]

[Postprocessors]
  [pp_inj]
    type = PointValue
    variable = 'pp'
    point = '500 0 0'
  []

  [pp_prod]
    type = PointValue
    variable = 'pp'
    point = '610 0 0'
  []

  [pp_mon1]
    type = PointValue
    variable = 'pp'
    point = '435 5 0'
  []

  [pp_mon2]
    type = PointValue
    variable = 'pp'
    point = '545 2 0'
  []

  [pp_mon3]
    type = PointValue
    variable = 'pp'
    point = '545 5 0'
  []
  [pp_mon4]
    type = PointValue
    variable = 'pp'
    point = '545 10 0'
  []

  [pp_mon5]
    type = PointValue
    variable = 'pp'
    point = '545 50 0'
  []

  [pp_mon6]
    type = PointValue
    variable = 'pp'
    point = '545 200 0'
  []

  [pp_mon7]
    type = PointValue
    variable = 'pp'
    point = '435 2 0'
  []

  [strain_yy_inj]
    type = PointValue
    variable = 'strain_yy'
    point = '500 0 0'
  []

  [diff_inj]
    type = PointValue
    variable = 'diffusivity'
    point = '500 0 0'
  []

  [strain_yy_prod]
    type = PointValue
    variable = 'strain_yy'
    point = '610 0 0'
  []
[]

[VectorPostprocessors]
  [LineSampler]
    type = LineValueSampler
    variable = 'pp strain_yy strain_xx'
    end_point = '625 0 0'
    start_point = '375 0 0'
    num_points = 500
    sort_by = x
  []

  [LineSampler_prod]
    type = LineValueSampler
    variable = 'pp strain_yy strain_xx'
    end_point = '610 500 0'
    start_point = '610 -500 0'
    num_points = 1000
    sort_by = y
  []

  [LineSampler_inj]
    type = LineValueSampler
    variable = 'pp strain_yy strain_xx'
    end_point = '500 500 0'
    start_point = '500 -500 0'
    num_points = 1000
    sort_by = y
  []
[]

[FluidProperties]
  [simple_fluid]
    type = SimpleFluidProperties
    bulk_modulus = 2.2E9
    viscosity = 1.0E-3
    density0 = 1000.0
    thermal_expansion = 0.0002
    cp = 4194
    cv = 4186
    porepressure_coefficient = 1
  []
[]

[Materials]
  [porosity_matrix]
    type = PorousFlowPorosityConst
    porosity = 0.01
    block = matrix
  []

  [porosity_fracture]
    type = PorousFlowPorosityConst
    porosity = 0.1
    block = fracture
  []

  [porosity_srv]
    type = PorousFlowPorosityConst
    porosity = 0.1
    block = 'srv srv2 srv3'
  []

  [temperature]
    type = PorousFlowTemperature
  []

  [permeability_matrix]
    type = PorousFlowPermeabilityConst
    permeability = '1E-20 0 0   0 1E-20 0   0 0 0'
    block = matrix
  []

  [permeability_fracture]
    type = PorousFlowPermeabilityConst
    permeability = '1E-12 0 0   0 1E-12 0   0 0 0'
    block = fracture
  []

  [permeability_srv]
    type = PorousFlowPermeabilityConst
    permeability = '1E-17 0 0   0 1E-17 0   0 0 0'
    block = 'srv srv2 srv3'
  []


  [biot_modulus]
    type = PorousFlowConstantBiotModulus
    biot_coefficient = 0.7
    solid_bulk_compliance = 2E-11
    fluid_bulk_modulus = 2.2E9
    block = 'matrix fracture srv srv2 srv3'
  []

  [massfrac]
    type = PorousFlowMassFraction
  []

  [simple_fluid]
    type = PorousFlowSingleComponentFluid
    fp = simple_fluid
    phase = 0
  []

  [PS]
    type = PorousFlow1PhaseFullySaturated
    porepressure = pp
  []

  [relp]
    type = PorousFlowRelativePermeabilityConst
    phase = 0
  []

  [eff_fluid_pressure_qp]
    type = PorousFlowEffectiveFluidPressure
  []

  [elasticity_tensor_matrix]
    type = ComputeIsotropicElasticityTensor
    poissons_ratio = 0.2
    bulk_modulus = 5.0E10
  []

  [strain]
    type = ComputeSmallStrain
    displacements = 'disp_x disp_y'
  []

  [stress]
    type = ComputeLinearElasticStress
  []

  [vol_strain]
    type = PorousFlowVolumetricStrain
  []

  #[hydraulic_diffusivity_frac]
  #  type = GenericFunctionMaterial
  #  prop_names = 'diffusivity_frac'
  #  prop_values = D_func_frac
  #  block = fracture
  #[]
  #[hydraulic_diffusivity_mat]
  #  type = GenericFunctionMaterial
  #  prop_names = 'diffusivity_mat'
  #  prop_values = D_func_mat
  #  block = matrix
  #[]
[]



[AuxVariables]
  [stress_xx]
    order = CONSTANT
    family = MONOMIAL
  []

  [stress_xy]
    order = CONSTANT
    family = MONOMIAL
  []

  [stress_yx]
    order = CONSTANT
    family = MONOMIAL
  []
  [stress_yy]
    order = CONSTANT
    family = MONOMIAL
  []

  [strain_xx]
    order = CONSTANT
    family = MONOMIAL
  []
  [strain_yy]
    order = CONSTANT
    family = MONOMIAL
  []
  [diffusivity]
    family = MONOMIAL
    order = CONSTANT
  []
[]

[AuxKernels]
  [stress_xx]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xx
    index_i = 0
    index_j = 0
  []
  [stress_xy]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xy
    index_i = 0
    index_j = 1
  []

  [stress_yx]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_yx
    index_i = 1
    index_j = 0
  []
  [stress_yy]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_yy
    index_i = 1
    index_j = 1
  []

  [strain_xx]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_xx
    index_i = 0
    index_j = 0
  []
  [strain_yy]
    type = RankTwoAux
    rank_two_tensor = total_strain
    variable = strain_yy
    index_i = 1
    index_j = 1
  []
  #[./set_diffusivity]
  #  type = FunctionAux
  #  variable = diffusivity
  #  function = diff_func
  #[../]
[]


[Preconditioning]
  active = mumps
  [mumps]
    type = SMP
    full = true
    petsc_options = '-snes_converged_reason -ksp_diagonal_scale -ksp_diagonal_scale_fix -ksp_gmres_modifiedgramschmidt -snes_linesearch_monitor'
    petsc_options_iname = '-ksp_type -pc_type -pc_factor_mat_solver_package -pc_factor_shift_type'
    petsc_options_value = 'gmres      lu       mumps                         NONZERO'
  []
  [basic]
    type = SMP
    full = true
  []
  [preferred_but_might_not_be_installed]
    type = SMP
    full = true
    petsc_options_iname = '-pc_type -pc_factor_mat_solver_package'
    petsc_options_value = ' lu       mumps'
  []
[]

[Functions]
  [./constant_step_1]
    type = PiecewiseConstant
    x = '204600 300000 310000 340000 350000 353000 353200 353400 353800 354000 354400 354600 354800  355000 355600 355800 356000 356100'
    y = '200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 100 100'
  [../]
  [./constant_step_2]
    type = PiecewiseConstant
    x = '356200 356400 356800 357000 357400 357600 357800 358000 358200 358400 358600 358800 359000 359100 359200'
    y = '100 200 200 200 200 200 200 200 200 200 200 100 100 100'
  [../]

  [./adaptive_step]
    type = PiecewiseConstant
    x = '359300 359400 360000 360400 360600 360800 361200 361600 362000 362400 362800 363000 363200 363400 363600 363700 363800 363900 364000'
    y =  '100 100 200 200 200 200 200 200 200 200 200 100 100 100 100 100 100 100 100'
  [../]

  [./adaptive_final]
    type = PiecewiseConstant
    x = '364200 364800 365200 365800 366400 366800 367400 367800 369000 370000 371000 372000 373000 374000 375000 376000 377000 377200 377400 377600 377800 378000 378200 378300 378400 378500 378600 378800 379000 380000 380400 381000 382000'
    y = '200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 200 100 100 100 100 100 100 100 100 200 200 200 200'
  [../]

  #[./adaptive_after]
  # type = PiecewiseConstant
  #  x = '382000 383000 384000 386000 388000 390000 392000 394000 396000 398000 400000 402000 404000 406000 408000 410000 412000 414000 416000 418000 420000 422000 424000 426000 428000 430000 432000 434000 436000 438000 440000 442000 444000 446000 448000 450000 452000 454000 456000 458000 460000 461221'
  #  y = '50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100'
  #[]
[../]


[Executioner]
  type = Transient
  solve_type = Newton
  end_time = 461221
  verbose = true
  [./TimeStepper]
    type = IterationAdaptiveDT
    dt = 200
    timestep_limiting_function = 'constant_step_1 constant_step_2 adaptive_step adaptive_final'
    force_step_every_function_point = true
 ## post_function_sync_dt = 0.1  # Adjust as necessary for stability
  [../]
  l_tol = 1e-3
  l_max_its = 2000
  nl_max_its = 200

  nl_abs_tol = 1e-3
  nl_rel_tol = 1e-3
[]

[Outputs]
  exodus=true
  [csv]
    type = CSV
  []
[]