"""Observation and geometry invariants; actual FE regressions live in the suite."""
import numpy as np
import pytest
from fiberis.moose.numerical_ddm import (Well,Element,rotation,axial_response,time_derivative,
    RectangularFracture,get_ellipse_fracture,NumericalConfig,write_input)


def test_oblique_affine_strain_and_rigid_motion():
    s=np.array([0.,.7,2.,3.,5.])
    tangent=np.array([2.,3.,4.]); tangent/=np.linalg.norm(tangent)
    p=s[:,None]*tangent
    A=np.array([[1.,2.,3.],[-1.,3.,5.],[2.,1.,4.]])*1e-4
    u=p@A.T+np.array([7.,4.,9.])
    _,_,eps=axial_response(p,u)
    np.testing.assert_allclose(eps,tangent@A@tangent,atol=2e-14)
    skew=A-A.T
    _,_,eps=axial_response(p,p@skew.T)
    np.testing.assert_allclose(eps,0.,atol=1e-17)


def test_curved_isotropic_dilation_gauge():
    theta=np.linspace(-.7,.7,201)
    p=np.column_stack([100*np.cos(theta),100*np.sin(theta),theta*30])
    _,_,eps=axial_response(p,p*.001,gauge_length=20)
    # Centered chord / arclength differs by O(ds^2); no spurious dt/ds dot u term.
    np.testing.assert_allclose(eps,.001,rtol=1e-5)


def test_nonuniform_time_quadratic():
    t=np.array([0.,1.,3.,6.])
    y=np.vstack([2+t**2,3*t-1])
    np.testing.assert_allclose(time_derivative(y,t),np.vstack([2*t,np.full(4,3.)]),atol=1e-14)
    with pytest.raises(ValueError): time_derivative(y,[0,1,1,3])


def test_rotation_and_sign():
    R=rotation(37,90)
    np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-15)
    np.testing.assert_allclose(np.linalg.det(R),1.)
    assert np.max(abs(R-rotation(37,90,legacy_vertical=True)))>1e-4
    np.testing.assert_array_equal(Element(100,60,.001,.002,-.003).jump,[.002,-.003,.001])


def test_dynamic_width_and_strip_geometry():
    f=RectangularFracture().define_asymmetric_grow([0,1,3],30,20,40,50,[.001,.002,.003])
    np.testing.assert_allclose([-x.elements[0].width for x in f.fractures],[.001,.002,.003])
    np.testing.assert_allclose(f.fractures[0].elements[0].center,[5,5,0])
    ell=get_ellipse_fracture(100,1.2,2e-5,0,0,5)
    assert len(ell.elements)==5
    assert ell.elements[2].height==60
    assert -ell.elements[2].width==.001


def test_conforming_input(tmp_path):
    c=NumericalConfig('/not/run',str(tmp_path),band_thickness=2,band_cells=4)
    path=write_input([Element(100,60,.001,.002,.003)],[[1,2,40],[1,3,40],[1,4,40]],c,tmp_path)
    text=path.read_text()
    assert 'StressDivergenceTensors' in text
    assert "eigen_base = '0 0 0.0005 0.00075 0.0005 0'" in text
    assert 'pressure' not in text
    assert 'disp_z' not in text  # Variables are ux,uy,uz, all independently solved.
