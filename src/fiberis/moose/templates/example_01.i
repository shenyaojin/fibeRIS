# The input file for the MOOSE framework. It is from input file generator in fibeRIS and the prototype is from
# https://mooseframework.inl.gov/getting_started/examples_and_tutorials/examples/ex01_inputfile.html
# Before you run the example, you may delete the comments in the file.
[Mesh]
  file = 'mug.e'
[]

[Variables]
  [./diffused]
    order = 'FIRST'
    family = 'LAGRANGE'
  []
[]

[Kernels]
  [./diff]
    type = Diffusion
    variable = 'diffused'
  []
[]

[BCs]
  [./bottom]
    type = DirichletBC
    variable = 'diffused'
    boundary = 'bottom'
    value = 1
  []
  [./top]
    type = DirichletBC
    variable = 'diffused'
    boundary = 'top'
    value = 0
  []
[]

[Executioner]
  type = 'Steady'
  solve_type = 'PJFNK'
[]

[Outputs]
  execute_on = 'timestep_end'
  exodus = true
[]