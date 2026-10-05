"""Three-dimensional prescribed DD via conforming elastic eigenstrain bands.

Coordinates, lengths and jumps: metres; E: Pa; time: seconds. The local
jump is u(0+) - u(0-) = (S1,S2,width). This regularization converges to a
Volterra rectangular dislocation as band thickness -> 0, away from edges.
No analytical displacement solution is used by this module.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from copy import deepcopy
import json
import time
import resource
import hashlib
import shlex
import re
from contextlib import redirect_stdout
import numpy as np
from .input_generator import MooseBlock
from .runner import MooseRunner


@dataclass
class NumericalConfig:
    executable: str
    output_directory: str
    youngs_modulus: float = 30e9
    poissons_ratio: float = 0.3
    half_extent: float = 800.0
    core_extent: float = 120.0
    core_extents: tuple[float,float,float] | None = None
    mesh_size: float = 20.0
    band_thickness: float = 2.0
    band_cells: int = 2
    growth: float = 1.5
    local_refinements: tuple = ()  # (axis, lower, upper, max_cell_size) in metres
    reuse_constant_jacobian: bool = False
    recycle_krylov_guess: bool = False
    max_estimated_dofs: int = 5_000_000
    order: str = 'SECOND'
    quadrature: str = 'FIFTH'
    linear_tolerance: float = 1e-10
    processors: int = 1
    mpiexec: str | None = None

    def __post_init__(self):
        if not (self.youngs_modulus > 0 and -1 < self.poissons_ratio < .5):
            raise ValueError('Invalid isotropic elastic constants')
        if not (0 < self.mesh_size and 0 < self.band_thickness and
                0 < self.core_extent < self.half_extent and self.growth > 1):
            raise ValueError('Invalid mesh/domain settings')
        if (not isinstance(self.band_cells,int) or self.band_cells < 1 or
                self.processors<1 or self.order not in ('FIRST', 'SECOND')):
            raise ValueError('Invalid band_cells/order')


class Element:
    def __init__(self, length, height, width, S1=0., S2=0.):
        if length <= 0 or height <= 0:
            raise ValueError('length and height must be positive')
        self.length, self.height = float(length), float(height)
        self.width, self.S1, self.S2 = -float(width), -float(S1), -float(S2)
        self.mu = .3
        self.set_global_coors(0., 0., 0.)

    def set_global_coors(self, x1, x2, x3):
        self.abs_x1, self.abs_x2, self.abs_x3 = map(float, (x1,x2,x3))
        return self

    @property
    def center(self):
        return np.array([self.abs_x1, self.abs_x2, self.abs_x3])

    @property
    def jump(self):
        return -np.array([self.S1, self.S2, self.width])


def rotation(strike, dip, isdeg=True, legacy_vertical=False):
    if isdeg:
        strike, dip = np.deg2rad([strike, dip])
    if legacy_vertical and dip == np.pi/2:
        dip -= 1e-3
    strike = np.pi-strike
    return np.array([[1,0,0],[0,np.cos(dip),np.sin(dip)],
                     [0,-np.sin(dip),np.cos(dip)]]) @ np.array([
                         [np.sin(strike),-np.cos(strike),0],
                         [np.cos(strike),np.sin(strike),0],[0,0,1]])


def axial_response(points, displacement, gauge_length=None, measured_depth=None, gauge_method='arclength'):
    """Centered chord tangent dot du/ds; repeat interior endpoints.

    Gauge response integrates this infinitesimal axial strain along measured
    depth using piecewise linear interpolation, with clipped endpoint windows.
    Tangent projection is outside the derivative (also for curved wells).
    """
    p, u = np.asarray(points, float), np.asarray(displacement, float)
    if p.shape != u.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError('Need matching (n>=3,3) points and displacement')
    ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if np.any(ds <= 0):
        raise ValueError('Consecutive well coordinates must be distinct')
    s = np.r_[0., np.cumsum(ds)] if measured_depth is None else np.asarray(measured_depth,float)
    if s.shape!=(len(p),) or not np.all(np.isfinite(s)) or np.any(np.diff(s)<=0):
        raise ValueError('measured_depth must be finite, strictly increasing and match the trajectory')
    chord = p[2:]-p[:-2]
    norms = np.linalg.norm(chord, axis=1)
    if np.any(norms == 0):
        raise ValueError('Well reverses direction over a centered stencil')
    tangent = chord/norms[:,None]
    strain = np.sum((u[2:]-u[:-2])*tangent, axis=1)/(s[2:]-s[:-2])
    strain = np.r_[strain[0],strain,strain[-1]]
    tangent = np.vstack([tangent[0],tangent,tangent[-1]])
    if gauge_length is not None:
        if gauge_length <= 0:
            raise ValueError('gauge_length must be positive')
        if gauge_method not in ('arclength','sample_mean'):
            raise ValueError('Unknown gauge method')
        averaged = []
        for si in s:
            lo, hi = max(s[0],si-gauge_length/2), min(s[-1],si+gauge_length/2)
            if gauge_method=='sample_mean':
                averaged.append(np.mean(strain[(s>=lo)&(s<=hi)]))
                continue
            knots = np.r_[lo,s[(s>lo)&(s<hi)],hi]
            values=np.interp(knots,s,strain)
            averaged.append(np.sum(.5*(values[1:]+values[:-1])*np.diff(knots))/(hi-lo))
        strain = np.array(averaged)
    return s, np.sum(u*tangent,axis=1), strain


def time_derivative(data, times):
    times = np.asarray(times,float)
    if len(times)<2 or np.any(np.diff(times)<=0) or np.shape(data)[-1]!=len(times):
        raise ValueError('Need >=2 strictly increasing times matching data last axis')
    return np.gradient(data,times,axis=-1,edge_order=2 if len(times)>2 else 1)


class Well:
    def __init__(self):
        self.gauge_length = None
        self.gauge_method = 'arclength'
        self.strain_calculation_method = 'projection'

    def set_well_path(self,x,y,z):
        self.x,self.y,self.z = [np.array(a,float,copy=True) for a in np.broadcast_arrays(x,y,z)]
        self.mds,_,_ = axial_response(self.points,np.zeros_like(self.points))
        return self

    @property
    def points(self):
        return np.column_stack([self.x,self.y,self.z])

    def calculate_strain(self):
        if self.strain_calculation_method != 'projection':
            raise NotImplementedError('Only infinitesimal axial projection is supported')
        self.mds,self.disp,self.strain = axial_response(
            self.points,np.column_stack([self.u1,self.u2,self.u3]),self.gauge_length,self.mds,self.gauge_method)

    calculate_displacement_along_well = calculate_strain


def _block(name, kind=None, **params):
    b = MooseBlock(name,kind)
    for k,v in params.items():
        if isinstance(v,(list,tuple,np.ndarray)):
            v = ' '.join(map(str,v))
        b.add_param(k,v)
    return b


def _axis_nodes(edges, config, axis=0):
    """Force every material interface onto element faces; grade only outside core."""
    core=config.core_extent if config.core_extents is None else config.core_extents[axis]
    if not 0<core<config.half_extent:
        raise ValueError('Each core extent must lie between zero and half_extent')
    if min(edges) <= -core or max(edges) >= core:
        raise ValueError('All bands must be strictly inside core_extent')
    windows=[(lo,hi,h) for k,lo,hi,h in config.local_refinements if k==axis]
    for lo,hi,h in windows:
        if not (-core<=lo<hi<=core and h>0):
            raise ValueError('Refinement windows must be ordered and inside the core')
    breaks = np.unique([-core,*edges,*[x for lo,hi,h in windows for x in (lo,hi)],core])
    values = []
    for a,b in zip(breaks[:-1],breaks[1:]):
        h=min([config.mesh_size]+[h for lo,hi,h in windows if lo<=(a+b)/2<=hi])
        values.extend(np.linspace(a,b,int(np.ceil((b-a)/h))+1)[:-1])
    values.append(core)
    outer, pos, step = [],core,config.mesh_size
    while pos < config.half_extent:
        pos = min(pos+step,config.half_extent)
        outer.append(pos)
        step *= config.growth
    return np.array([-x for x in outer[::-1]]+values+outer)


def write_input(elements, points, config, directory, load_history=None):
    """Self-contained input, homogeneous 3D elasticity, zero remote displacement."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    p = np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or not np.all(np.isfinite(p)):
        raise ValueError('Receivers must be a finite (n,3) array')
    if np.any(abs(p)>=config.half_extent):
        raise ValueError('Receivers must lie strictly inside the domain')
    edges = [[],[],[]]
    for e in elements:
        if not np.isclose(e.mu,config.poissons_ratio):
            raise ValueError('Element.mu must equal configured poissons_ratio')
        for k,span in enumerate([e.length,e.height,config.band_thickness]):
            edges[k].extend(e.center[k]+np.linspace(-span/2,span/2,
                           config.band_cells+1 if k==2 else 2))
    axes = [_axis_nodes(edge,config,k) for k,edge in enumerate(edges)]
    degree=2 if config.order=='SECOND' else 1
    estimated_dofs=3*int(np.prod([degree*(len(a)-1)+1 for a in axes]))
    if estimated_dofs>config.max_estimated_dofs:
        raise ValueError(f'Estimated {estimated_dofs:,} DOFs exceeds configured resource cap '
                         f'{config.max_estimated_dofs:,}; coarsen mesh/core or explicitly raise the cap.')
    mesh = _block('Mesh',use_displaced_mesh=False,parallel_type='DISTRIBUTED')
    mesh.add_sub_block(_block('cartesian','CartesianMeshGenerator',dim=3,
                             **{k:np.diff(axis) for k,axis in zip(['dx','dy','dz'],axes)}))
    mesh.add_sub_block(_block('translate','TransformGenerator',input='cartesian',
                             transform='TRANSLATE',vector_value=[-config.half_extent]*3))
    if config.order == 'SECOND':
        mesh.add_sub_block(_block('quadratic','ElementOrderConversionGenerator',
                                 input='translate',conversion_type='SECOND_ORDER'))
    variables = _block('Variables')
    kernels, bcs = _block('Kernels'), _block('BCs')
    disp = ['ux','uy','uz']
    for i,v in enumerate(disp):
        variables.add_sub_block(_block(v,order=config.order))
        kernels.add_sub_block(_block(v,'StressDivergenceTensors',variable=v,component=i))
        bcs.add_sub_block(_block(v,'DirichletBC',variable=v,boundary='left right top bottom front back',value=0))
    materials, functions = _block('Materials'),_block('Functions')
    if load_history is not None:
        times,factors=[np.asarray(a,float) for a in load_history]
        if (times.ndim!=1 or factors.shape!=times.shape or len(times)<2 or
                not np.all(np.isfinite(times)) or not np.all(np.isfinite(factors)) or
                np.any(np.diff(times)<=0) or factors[0]!=0):
            raise ValueError('Fixed-geometry load history requires increasing finite times and a zero initial load')
        functions.add_sub_block(_block('load_history','PiecewiseLinear',x=times,y=factors))
    eigen_names=[]
    for i,e in enumerate(elements):
        name=f'band_{i}'
        box='*'.join(f'(abs({v}-({c:.17g}))<{span/2:.17g})'
                     for v,c,span in zip('xyz',e.center,[e.length,e.height,config.band_thickness]))
        spatial_name=name if load_history is None else name+'_space'
        functions.add_sub_block(_block(spatial_name,'ParsedFunction',expression=box))
        if load_history is not None:
            functions.add_sub_block(_block(name,'CompositeFunction',functions=[spatial_name,'load_history']))
        materials.add_sub_block(_block(name+'_mask','GenericFunctionMaterial',prop_names=name+'_mask',prop_values=name))
        j=e.jump/config.band_thickness
        materials.add_sub_block(_block(name,'ComputeEigenstrain',eigenstrain_name=name,
                                       eigen_base=[0,0,j[2],j[1]/2,j[0]/2,0],prefactor=name+'_mask'))
        eigen_names.append(name)
    materials.add_sub_block(_block('strain','ComputeSmallStrain',eigenstrain_names=eigen_names))
    materials.add_sub_block(_block('elasticity','ComputeIsotropicElasticityTensor',
                                   youngs_modulus=config.youngs_modulus,poissons_ratio=config.poissons_ratio))
    materials.add_sub_block(_block('stress','ComputeLinearElasticStress'))
    sampler=_block('VectorPostprocessors')
    sampler.add_sub_block(_block('receivers','PointValueSampler',variable=disp,
                                points=p.ravel(),sort_by='id',execute_on='TIMESTEP_END' if load_history is None else 'INITIAL TIMESTEP_END'))
    precond=_block('Preconditioning')
    precond.add_sub_block(_block('full','SMP',full=True))
    executioner=_block('Executioner','Steady',solve_type='NEWTON',line_search='none',
                       nl_rel_tol=1e-9,nl_abs_tol=1e-8,nl_max_its=5,l_tol=config.linear_tolerance,
                       l_max_its=3000,petsc_options_iname='-ksp_type -pc_type -pc_hypre_type',
                       petsc_options_value='gmres hypre boomeramg')
    if load_history is not None:
        executioner.block_type='Transient'
        for key,value in dict(start_time=float(times[0]),end_time=float(times[-1]),
                              abort_on_solve_fail=True).items():
            executioner.add_param(key,value)
        executioner.add_sub_block(_block('TimeStepper','TimeSequenceStepper',time_sequence=times))
        # Fixed mesh + linear elastic moduli: tangent stiffness is constant.
        # PETSc still solves equilibrium for every requested load; no analytic
        # displacement or externally scaled solution is supplied.
        if config.reuse_constant_jacobian:
            executioner.add_param('reuse_preconditioner',True)
            executioner.add_param('reuse_preconditioner_max_linear_its',1000)
        if config.recycle_krylov_guess:
            # PETSc builds an initial linear-solver guess from earlier FE solves.
            # The original RHS, KSP tolerance and nonlinear residual checks stay
            # unchanged. No analytical/reference field enters this mechanism.
            executioner.add_param('petsc_options_iname','-ksp_type -pc_type -pc_hypre_type -ksp_fischer_guess')
            executioner.add_param('petsc_options_value','gmres hypre boomeramg 1,5')
    blocks=[mesh,_block('GlobalParams',displacements=disp),variables,kernels,bcs,
            functions,materials,precond,executioner,_block('Problem','ConstantJacobianProblem' if load_history is not None and config.reuse_constant_jacobian else None),sampler,
            _block('Outputs',csv=True,file_base='solution',perf_graph=True)]
    executioner.add_sub_block(_block('Quadrature',order=config.quadrature))
    path=directory/'model.i'
    path.write_text('# 3D FE prescribed DD; SI; no analytic boundary data.\n'+
                    '\n\n'.join(b.render()+'\n[]' for b in blocks)+'\n')
    (directory/'model.json').write_text(json.dumps(dict(config=asdict(config),
        elements=[vars(e) for e in elements],points=p.tolist(),
        cells=[len(a)-1 for a in axes],
        load_history=None if load_history is None else dict(times=times.tolist(),factors=factors.tolist())),indent=2))
    return path


class FractureZ:
    def __init__(self):
        self.elements=[]
        self.monitor_wells=[]
        self.config=None

    def set_fracture_elements(self,elements):
        self.elements=elements if isinstance(elements,list) else [elements]
        return self

    def set_monitor_wells(self,wells):
        self.monitor_wells=deepcopy(wells if isinstance(wells,list) else [wells])
        return self

    def configure(self,config):
        self.config=config
        return self

    def frame(self):
        return np.eye(3),np.zeros(3)

    def sample(self,points,config=None,load_history=None):
        config=config or self.config
        if config is None:
            raise ValueError('Call configure(NumericalConfig(...)) before calculate')
        if not self.elements:
            raise ValueError('No fracture elements')
        R,origin=self.frame()
        local=(np.asarray(points,float)-origin)@R.T
        path=write_input(self.elements,local,config,config.output_directory,load_history)
        (path.parent/'global_frame.json').write_text(json.dumps(dict(
            global_to_local_rotation=R.tolist(),global_origin_m=origin.tolist(),
            global_receiver_points_m=np.asarray(points,float).tolist(),
            wells=[dict(measured_depth_m=w.mds.tolist(),gauge_length_m=w.gauge_length,
                        gauge_method=w.gauge_method) for w in self.monitor_wells]),indent=2))
        runner=MooseRunner(config.executable,config.mpiexec)
        # A rerun may request fewer time steps. Remove only our generated
        # receiver tables so a previous run cannot supply a stale final frame.
        for previous in path.parent.glob('solution_receivers_*.csv'):
            if re.match(r'solution_receivers_\d+\.csv$',previous.name):
                previous.unlink()
        for name in ('displacement.npz','displacement_history.npz'):
            (path.parent/name).unlink(missing_ok=True)
        (path.parent/'run.json').write_text(json.dumps(dict(
            success=False,status='running',input=str(path),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
        usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
        t0=time.monotonic()
        with (path.parent/'progress.log').open('w',buffering=1) as progress, redirect_stdout(progress):
            ok,out,err=runner.run(str(path),num_processors=config.processors,
                                  stream_output=True,clean_output_dir=False)
        elapsed=time.monotonic()-t0
        usage1=resource.getrusage(resource.RUSAGE_CHILDREN)
        command=([runner.mpiexec_path or "mpiexec","-n",str(config.processors)] if config.processors>1 else []) + [runner.moose_executable_path,"-i","model.i","--n-threads=1"]
        (path.parent/"command.txt").write_text(shlex.join(command)+"\n")
        (path.parent/'stderr.log').write_text(err)
        (path.parent/'run.json').write_text(json.dumps(dict(success=ok,elapsed_seconds=elapsed,
                   processors=config.processors,executable=runner.moose_executable_path,
                   input=str(path),returncode=runner.last_run_returncode,
                   input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   cpu_seconds=usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime,
                   child_maxrss_kib=usage1.ru_maxrss,
                   executable_sha256=hashlib.sha256(Path(runner.moose_executable_path).read_bytes()).hexdigest(),
                   command=command),indent=2))
        if not ok:
            raise RuntimeError(f'MOOSE failed: {path.parent}/log.txt and stderr.log')
        files=sorted(f for f in path.parent.glob('solution_receivers_*.csv')
                     if re.match(r'solution_receivers_\d+\.csv$',f.name))
        if not files:
            raise RuntimeError('MOOSE produced no receiver output')
        selected=[files[-1]] if load_history is None else files
        if load_history is not None and len(selected)!=len(load_history[0]):
            raise RuntimeError('MOOSE history count differs from the requested time grid')
        values=[]
        for file in selected:
            data=np.atleast_1d(np.genfromtxt(file,delimiter=',',names=True))
            data=np.sort(data,order='id')
            if len(data)!=len(local) or not np.allclose(np.column_stack([data[v] for v in 'xyz']),local,rtol=0,atol=1e-8):
                raise RuntimeError('Receiver coordinates/order mismatch')
            values.append(np.column_stack([data[v] for v in ['ux','uy','uz']])@R)
        if load_history is None:
            u=values[0]
            np.savez(path.parent/'displacement.npz',points=points,displacement=u)
        else:
            u=np.array(values)
            np.savez(path.parent/'displacement_history.npz',points=points,displacement=u,
                     times=load_history[0],factors=load_history[1])
        return u

    def sample_history(self,points,times,factors,config=None):
        """Solve each equilibrium in one MOOSE run for fixed geometry and scaled jumps.

        All elements/modes share factors(t); use DynamicFracture snapshots for
        general independent histories or changing geometry. Initial load must be
        zero. No inertia, diffusion, analytical fields or solution scaling.
        """
        return self.sample(points,config,load_history=(times,factors))

    def calculate_well_displacement(self):
        counts=[len(w.x) for w in self.monitor_wells]
        u=self.sample(np.vstack([w.points for w in self.monitor_wells]))
        for w,part in zip(self.monitor_wells,np.split(u,np.cumsum(counts)[:-1])):
            w.u1,w.u2,w.u3=part.T
            w.calculate_strain()
        return self

    calculate=calculate_well_displacement


class FractureGlobal(FractureZ):
    def __init__(self):
        super().__init__()
        self.strike=self.dip=90.
        self.fracx=self.fracy=self.fracz=0.
        self.isdeg=True
        self.legacy_vertical=False

    def set_global_coors(self,strike,dip,fracx,fracy,fracz):
        self.strike,self.dip,self.fracx,self.fracy,self.fracz=strike,dip,fracx,fracy,fracz
        return self

    def frame(self):
        return rotation(self.strike,self.dip,self.isdeg,self.legacy_vertical),np.array([self.fracx,self.fracy,self.fracz])


def fractureZ_to_global(frac,strike,dip,fracx,fracy,fracz,isdeg=True):
    f=FractureGlobal().set_global_coors(strike,dip,fracx,fracy,fracz).set_fracture_elements(frac.elements)
    f.isdeg=isdeg
    return f


def get_ellipse_fracture(length,height_ratio,width_ratio,S1_ratio,S2_ratio,element_N):
    dx=length/element_N
    elements=[]
    for x in np.arange(element_N)*dx-length/2+dx/2:
        r=np.sqrt(length**2/4-x*x)
        elements.append(Element(dx,r*height_ratio,r*width_ratio,r*S1_ratio,r*S2_ratio).set_global_coors(x,0,0))
    return FractureZ().set_fracture_elements(elements)


class DynamicFracture:
    """Sequence of quasi-static snapshots, with no inertial/time integration model."""
    def __init__(self):
        self.fractures=[]
        self.monitor_wells=[]
        self.taxis=None
        self.config=None

    def set_fractures(self,fractures):
        self.fractures=fractures if isinstance(fractures,list) else [fractures]
        return self

    def set_monitor_wells(self,wells):
        self.monitor_wells=deepcopy(wells if isinstance(wells,list) else [wells])
        return self

    def configure(self,config):
        self.config=config
        return self

    def calculate(self):
        from dataclasses import replace
        t=np.asarray(self.taxis,float)
        if t.ndim!=1 or len(t)!=len(self.fractures) or np.any(np.diff(t)<=0):
            raise ValueError('Strictly increasing taxis must match fracture snapshots')
        if self.config is None:
            raise ValueError('Configure the numerical backend first')
        for i,f in enumerate(self.fractures):
            c=replace(self.config,output_directory=str(Path(self.config.output_directory)/f'snapshot_{i:04d}'))
            f.set_monitor_wells(self.monitor_wells).configure(c).calculate()
        return self

    def gather_strain_data(self,reference_initial=False):
        from fiberis.analyzer.Data2D.core2D import Data2D
        result=[]
        for i,w in enumerate(self.monitor_wells):
            strain=np.column_stack([f.monitor_wells[i].strain for f in self.fractures])
            if reference_initial:
                strain=strain-strain[:,:1]
            item=Data2D(data=strain,taxis=np.asarray(self.taxis,float),daxis=w.mds.copy())
            item.x,item.y,item.z=w.x,w.y,w.z
            result.append(item)
        return result

    def gather_strain_rate_data(self):
        data=self.gather_strain_data()
        for item in data:
            item.data=time_derivative(item.data,self.taxis)
        return data


def _history(value,n):
    a=np.asarray(value if value is not None else 0.,float)
    return np.broadcast_to(a,(n,)).copy()


class RectangularFracture(DynamicFracture):
    def define_LHW(self,taxis,length,heigth,width=None,S1=None,S2=None):
        self.taxis=np.asarray(taxis,float)
        arrays=[_history(v,len(taxis)) for v in [length,heigth,width,S1,S2]]
        self.fractures=[FractureZ().set_fracture_elements(Element(*args)) for args in zip(*arrays)]
        return self

    def define_by_length(self,taxis,length,heigth,width,S1=0,S2=0):
        return self.define_LHW(taxis,length,heigth,width,S1,S2)

    length_growth=define_by_length

    def define_asymmetric_grow(self,taxis,up,down,left,right,width=None,S1=None,S2=None):
        n=len(taxis)
        up,down,left,right,width,S1,S2=[_history(v,n) for v in [up,down,left,right,width,S1,S2]]
        self.define_LHW(taxis,left+right,up+down,width,S1,S2)
        for i,f in enumerate(self.fractures):
            f.elements[0].set_global_coors((right[i]-left[i])/2,(up[i]-down[i])/2,0)
        return self


class EllipticalFracture(DynamicFracture):
    def define_by_length(self,taxis,length,height_ratio,width_ratio,S1_ratio=0,S2_ratio=0,element_N=10):
        self.taxis=np.asarray(taxis,float)
        self.fractures=[get_ellipse_fracture(L,height_ratio,width_ratio,S1_ratio,S2_ratio,element_N)
                        for L in _history(length,len(taxis))]
        return self


def from_ddmpy(fracture,config,length_scale=1.):
    """Import existing DDMPY geometry without importing its analytical evaluator.

    Original DDMPY stores negative jumps. Its mu is Poisson's ratio. For global
    fractures the legacy 90 degree perturbation is explicitly retained by this
    import adapter; new numerical fractures use the exact supplied orientation.
    """
    if not np.isfinite(length_scale) or length_scale<=0:
        raise ValueError('length_scale must be positive')
    if hasattr(fracture,'fractures'):
        result=DynamicFracture().configure(config)
        result.taxis=np.array(fracture.taxis,float)
        result.fractures=[from_ddmpy(f,config,length_scale) for f in fracture.fractures]
        wells=fracture.monitor_wells
    else:
        result=FractureGlobal() if hasattr(fracture,'strike') else FractureZ()
        if isinstance(result,FractureGlobal):
            for key in ['strike','dip','fracx','fracy','fracz','isdeg']:
                setattr(result,key,getattr(fracture,key))
            result.fracx*=length_scale
            result.fracy*=length_scale
            result.fracz*=length_scale
            result.legacy_vertical=True
        for original in fracture.elements:
            e=Element(*(length_scale*np.array([original.length,original.height,-original.width,-original.S1,-original.S2])))
            e.mu=original.mu
            e.set_global_coors(*(length_scale*np.array([original.abs_x1,original.abs_x2,original.abs_x3])))
            result.elements.append(e)
        result.configure(config)
        wells=getattr(fracture,'global_monitor_wells',None) or fracture.monitor_wells
    if wells:
        converted=[]
        for original in wells:
            w=Well().set_well_path(*(length_scale*np.array([original.x,original.y,original.z])))
            w.mds=length_scale*np.array(original.mds)
            w.gauge_length=None if original.gauge_length is None else original.gauge_length*length_scale
            w.gauge_method="sample_mean"
            w.strain_calculation_method=original.strain_calculation_method
            converted.append(w)
        result.set_monitor_wells(converted)
    return result


class Model:
    """Linear superposition of actual FE solves in independent rotated domains.

    The error is a sum of each finite-domain/regularization error. No mechanical
    interaction law is needed because each dislocation jump is prescribed.
    """
    def __init__(self,fractures,config):
        self.fractures=list(fractures)
        self.config=config

    def sample(self,points):
        from dataclasses import replace
        result=np.zeros_like(np.asarray(points,float))
        for i,f in enumerate(self.fractures):
            c=replace(self.config,output_directory=str(Path(self.config.output_directory)/f'fracture_{i:04d}'))
            result+=f.sample(points,c)
        return result
