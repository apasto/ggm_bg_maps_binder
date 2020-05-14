#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TO DO: description (in a hurry!)
"""

from pathlib import Path
import configparser
import numpy as np
import pyshtools
import shtaper
import pygmt
import copy

GMTMapProjection = 'Mollweide'
GMTMapRegion = 'g'  # 'g' global, region option of SHGrid.plotgmt()
GMTColormap = 'haxby'
FigDPI = 200  # default figure resolution

# %% paths, data input
DataPathsConfig = configparser.ConfigParser()
DataPathsConfig.read('DataPaths.cfg')

STORAGE_DIR = Path(DataPathsConfig['DataPaths']['STORAGE_DIR'])
DATA_DIR = STORAGE_DIR / "FastGGM_20200508/"
FIG_DIR = STORAGE_DIR / "OutFig/FastGGM_20200508/"

# %% load GGM gfc
SH_lmax_GOCO06s = 300
SH_lmax_high = 2159  # XGM and Earth2014
grid_step = 0.25
SH_lmax_grid = int(90/grid_step - 1)  # to get a grid_step spaced grid
grid_step_high = 90 / (SH_lmax_high + 1)
GOCO06s_cilm, GOCO06s_gm, GOCO06s_r0 = \
    pyshtools.shio.read_icgem_gfc(
        DATA_DIR / '../OutFig/GOCO06s.gfc', lmax=SH_lmax_GOCO06s)
XGM_cilm, XGM_gm, XGM_r0 = \
    pyshtools.shio.read_icgem_gfc(
        DATA_DIR / 'XGM2019e_2159.gfc', lmax=SH_lmax_high)
E2014_cilm, E2014_gm, E2014_r0 = \
    pyshtools.shio.read_icgem_gfc(
        DATA_DIR / 'dV_ELL_Earth2014.gfc', lmax=SH_lmax_high)

# GGM to SHGravCoeffs class
GOCO06s_coeffs = pyshtools.SHGravCoeffs.from_array(
    GOCO06s_cilm, GOCO06s_gm, GOCO06s_r0,
    omega=pyshtools.constant.omega_wgs84.value,
    lmax=SH_lmax_GOCO06s)
XGM_coeffs = pyshtools.SHGravCoeffs.from_array(
    XGM_cilm, XGM_gm, XGM_r0,
    omega=pyshtools.constant.omega_wgs84.value,
    lmax=SH_lmax_high)
E2014_coeffs = pyshtools.SHGravCoeffs.from_array(
    E2014_cilm, E2014_gm, E2014_r0,
    omega=0,
    lmax=SH_lmax_high)

# %% apply gentle-cut low pass of E2014 to sat-resolution grid
LP_weights = \
    shtaper.taper_weights(l_start=180, l_stop=300,
                          l_max=SH_lmax_high, taper="gentle")
E2014_coeffs_LP = copy.deepcopy(E2014_coeffs)
E2014_coeffs_LP.coeffs = np.multiply(E2014_coeffs_LP.coeffs, LP_weights)
E2014_coeffs_LP.pad(lmax=SH_lmax_grid, copy=False)

# %% strange ringing in XGM2019 observed when zooming in, at mid-high latitudes: applying gentle-cut low pass
# applying also to E2014
LP_weights_XGM = \
    shtaper.taper_weights(l_start=1850, l_stop=2150,
                          l_max=SH_lmax_high, taper="gentle")
XGM_coeffs_LP = copy.deepcopy(XGM_coeffs)
XGM_coeffs_LP.coeffs = np.multiply(XGM_coeffs_LP.coeffs, LP_weights_XGM)
E2014_coeffs_LP_high = copy.deepcopy(E2014_coeffs)
E2014_coeffs_LP_high.coeffs = np.multiply(E2014_coeffs_LP_high.coeffs, LP_weights_XGM)

# %% expand to SHGravGrid, then to SHGrid (to get the plotgmt() method), convert to mGal
GOCO06s_grid = pyshtools.SHGravCoeffs.expand(
    GOCO06s_coeffs,
    a=GOCO06s_r0, f=pyshtools.constant.f_wgs84.value,
    extend=False, lmax=SH_lmax_grid, normal_gravity=True)
GOCO06s_grid_gd = pyshtools.SHGrid.from_array(GOCO06s_grid.total.data*1e5)
XGM_grid = pyshtools.SHGravCoeffs.expand(
    XGM_coeffs_LP,
    a=XGM_r0, f=pyshtools.constant.f_wgs84.value,
    extend=False, lmax=SH_lmax_high, normal_gravity=True)
XGM_grid_gd = pyshtools.SHGrid.from_array(XGM_grid.total.data*1e5)
E2014_grid = pyshtools.SHGravCoeffs.expand(
    E2014_coeffs_LP_high,
    a=E2014_r0, f=pyshtools.constant.f_wgs84.value,
    extend=False, lmax=SH_lmax_high, normal_gravity=False)
E2014_grid_rad = pyshtools.SHGrid.from_array(E2014_grid.rad.data * -1e5)  # to mGal and positive downwards
E2014_grid_low = pyshtools.SHGravCoeffs.expand(
    E2014_coeffs_LP,
    a=E2014_r0, f=pyshtools.constant.f_wgs84.value,
    extend=False, lmax=SH_lmax_grid, normal_gravity=False)
E2014_grid_low_rad = pyshtools.SHGrid.from_array(E2014_grid_low.rad.data * -1e5)  # to mGal and positive downwards

# %% apply terrain correction 'the proper way'
# perform difference of potential (adimensional) grid, then expand to SH coefficients, then to gravity grids
# normal gravity is removed after applying the reduction
# (potential issue: what if gm and r0 are different between GGM and terrain effect model?)

# GOCO06s, low SH degrees
GOCO06s_PotGrid_adim = (pyshtools.SHCoeffs.from_array(GOCO06s_cilm)).expand()
E2014_PotGrid_adim_low = (pyshtools.SHCoeffs.from_array(E2014_cilm)).expand(lmax=GOCO06s_PotGrid_adim.lmax)
GOCO06s_Vbg_Coeffs = (pyshtools.SHGrid.from_array(GOCO06s_PotGrid_adim.data - E2014_PotGrid_adim_low.data)).expand()
GOCO06s_Vbg_GravCoeffs = pyshtools.SHGravCoeffs.from_array(GOCO06s_Vbg_Coeffs.coeffs,
                                                           gm=GOCO06s_gm, r0=GOCO06s_r0,
                                                           omega=pyshtools.constant.omega_wgs84.value)
GOCO06s_Vbg_GravGrid = GOCO06s_Vbg_GravCoeffs.expand(normal_gravity=True,
                                                     a=GOCO06s_r0, f=pyshtools.constant.f_wgs84.value)
GOCO06s_Vbg_GravGrid_total = pyshtools.SHGrid.from_array(GOCO06s_Vbg_GravGrid.total.data * 1e5)

# XGM2019, high SH degrees
XGM_PotGrid_adim = (pyshtools.SHCoeffs.from_array(XGM_cilm)).expand()
E2014_PotGrid_adim = (pyshtools.SHCoeffs.from_array(E2014_cilm)).expand(lmax=XGM_PotGrid_adim.lmax)
XGM_Vbg_Coeffs = (pyshtools.SHGrid.from_array(XGM_PotGrid_adim.data - E2014_PotGrid_adim.data)).expand()
XGM_Vbg_GravCoeffs = pyshtools.SHGravCoeffs.from_array(XGM_Vbg_Coeffs.coeffs,
                                                       gm=XGM_gm, r0=XGM_r0,
                                                       omega=pyshtools.constant.omega_wgs84.value)
XGM_Vbg_GravGrid = XGM_Vbg_GravCoeffs.expand(normal_gravity=True,
                                              a=XGM_r0, f=pyshtools.constant.f_wgs84.value)
XGM_Vbg_GravGrid_total = pyshtools.SHGrid.from_array(XGM_Vbg_GravGrid.total.data * 1e5)

# %% plot maps (global)
cmap_gd_range = [-120, 120]  # fixed, to do: symmetric, largest value
cmap_tc_range = [np.quantile(E2014_grid_low_rad.data, 0.0001),
                 np.quantile(E2014_grid_low_rad.data, 0.9999)]
cmap_bg_range = [np.quantile(GOCO06s_Vbg_GravGrid_total.data, 0.001),
                 np.quantile(GOCO06s_Vbg_GravGrid_total.data, 0.999)]

gd_cpt_filename = str(FIG_DIR / 'ggm_gd.cpt')
tc_cpt_filename = str(FIG_DIR / 'ggm_tc.cpt')
bg_cpt_filename = str(FIG_DIR / 'ggm_bg.cpt')
pygmt.makecpt(series=cmap_gd_range, cmap='haxby', reverse=False, continuous=True, D='o', H=gd_cpt_filename)
pygmt.makecpt(series=cmap_tc_range, cmap='haxby', reverse=False, continuous=True, D='o', H=tc_cpt_filename)
pygmt.makecpt(series=cmap_bg_range, cmap='haxby', reverse=False, continuous=True, D='o', H=bg_cpt_filename)

GOCO06s_grid_gd_map = GOCO06s_grid_gd.plotgmt(
    projection=GMTMapProjection,
    region=GMTMapRegion,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=gd_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_gd_range,
    cb_triangles='both',
    cb_label='gravity disturbance [mGal]')
GOCO06s_grid_gd_map.coast(D='i', W='1/faint', t=50)
GOCO06s_grid_gd_map.basemap(B='0g15', t='75')
GOCO06s_grid_gd_map.savefig(FIG_DIR / 'GOCO06_300_GD.png', dpi=FigDPI)

XGM_grid_gd_map = XGM_grid_gd.plotgmt(
    projection=GMTMapProjection,
    region=GMTMapRegion,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=gd_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_gd_range,
    cb_triangles='both',
    cb_label='gravity disturbance [mGal]')
XGM_grid_gd_map.coast(D='i', W='1/faint', t=50)
XGM_grid_gd_map.basemap(B='0g15', t='75')
XGM_grid_gd_map.savefig(FIG_DIR / 'XGM2019_2159_GD.png', dpi=FigDPI)

E2014_grid_rad_map = E2014_grid_rad.plotgmt(
    projection=GMTMapProjection,
    region=GMTMapRegion,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=tc_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_tc_range,
    cb_triangles='both',
    cb_label='terrain effect, radial [mGal]')
E2014_grid_rad_map.coast(D='i', W='1/faint', t=50)
E2014_grid_rad_map.basemap(B='0g15', t='75')
E2014_grid_rad_map.savefig(FIG_DIR / 'E2014_2159_rad.png', dpi=FigDPI)

E2014_grid_low_rad_map = E2014_grid_low_rad.plotgmt(
    projection=GMTMapProjection,
    region=GMTMapRegion,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=tc_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_tc_range,
    cb_triangles='both',
    cb_label='terrain effect, radial [mGal]')
E2014_grid_low_rad_map.coast(D='i', W='1/faint', t=50)
E2014_grid_low_rad_map.basemap(B='0g15', t='75')
E2014_grid_low_rad_map.savefig(FIG_DIR / 'E2014_300_rad.png', dpi=FigDPI)

GOCO06s_grid_bg_map = GOCO06s_Vbg_GravGrid_total.plotgmt(
    projection=GMTMapProjection,
    region=GMTMapRegion,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=bg_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_bg_range,
    cb_triangles='both',
    cb_label='Bouguer disturbance [mGal]')
GOCO06s_grid_bg_map.coast(D='i', W='1/faint', t=50)
GOCO06s_grid_bg_map.basemap(B='0g15', t='75')
GOCO06s_grid_bg_map.savefig(FIG_DIR / 'GOCO06_300_BG.png', dpi=FigDPI)

XGM_grid_bg_map = XGM_Vbg_GravGrid_total.plotgmt(
    projection=GMTMapProjection,
    region=GMTMapRegion,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=bg_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_bg_range,
    cb_triangles='both',
    cb_label='Bouguer disturbance [mGal]')
XGM_grid_bg_map.coast(D='i', W='1/faint', t=50)
XGM_grid_bg_map.basemap(B='0g15', t='75')
XGM_grid_bg_map.savefig(FIG_DIR / 'XGM2019_2159_BG.png', dpi=FigDPI)

# %% Europe: zoomed in maps
EuropeZoom_extents = [0, 30, 30, 60]  # [West, East, South, North], region option of SHGrid.plotgmt()
EuropeZoom_central_latitude = (EuropeZoom_extents[3]+EuropeZoom_extents[2])/2
EuropeZoom_central_longitude = (EuropeZoom_extents[0]+EuropeZoom_extents[1])/2
cmap_bg_EuropeZoom_range = [-80, 380]
bg_EuropeZoom_cpt_filename = str(FIG_DIR / 'ggm_bg.cpt')
pygmt.makecpt(series=cmap_bg_EuropeZoom_range, cmap='haxby', reverse=False, continuous=True,
              D='o', H=bg_EuropeZoom_cpt_filename)

GOCO06s_grid_bg_map_EuropeZoom = GOCO06s_Vbg_GravGrid_total.plotgmt(
    projection='CYLindrical-stereographic',
    central_latitude=EuropeZoom_central_latitude,
    central_longitude=EuropeZoom_central_longitude,
    region=EuropeZoom_extents,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=bg_EuropeZoom_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_bg_EuropeZoom_range,
    cb_triangles='both',
    cb_label='Bouguer disturbance [mGal]')
GOCO06s_grid_bg_map_EuropeZoom.coast(D='i', W='1/thin', t=50)
GOCO06s_grid_bg_map_EuropeZoom.basemap(B='0g5', t='50')
GOCO06s_grid_bg_map_EuropeZoom.basemap(B=['5g0', 'WesN'])
GOCO06s_grid_bg_map_EuropeZoom.savefig(FIG_DIR / 'EuropeZoom_GOCO06_300_BG.png', dpi=FigDPI)

XGM_grid_bg_map_EuropeZoom = XGM_Vbg_GravGrid_total.plotgmt(
    projection='CYLindrical-stereographic',
    central_latitude=EuropeZoom_central_latitude,
    central_longitude=EuropeZoom_central_longitude,
    region=EuropeZoom_extents,
    grid=[0, 0], tick_interval=[0, 0], ticks='wesn',
    colorbar='bottom',
    cmap=bg_EuropeZoom_cpt_filename,
    cmap_continuous=True,
    cmap_limits=cmap_bg_EuropeZoom_range,
    cb_triangles='both',
    cb_label='Bouguer disturbance [mGal]')
XGM_grid_bg_map_EuropeZoom.coast(D='i', W='1/thin', t=50)
XGM_grid_bg_map_EuropeZoom.basemap(B='0g5', t='50')
XGM_grid_bg_map_EuropeZoom.basemap(B=['5g0', 'WesN'])
XGM_grid_bg_map_EuropeZoom.savefig(FIG_DIR / 'EuropeZoom_XGM2019_2159_BG.png', dpi=FigDPI)
