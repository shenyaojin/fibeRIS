# Using new lib compared to 101l
# Run in Lakora
# Shenyao Jin, 02112025
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fiberis.analyzer.Data1D import Data1D_PumpingCurve, Data1D_Gauge
from fiberis.analyzer.Data2D import Data2D_XT_DSS

#%% Define area of interest
stage1 = 7
stage2 = 8

coeff = 0.6
datapath = "data/fiberis_format/"

#%% Load pumping data (stage 7 and stage 8)
pc_data_folder_stg7 = datapath + "prod/pumping_data/stage7/"
pc_stg7_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg7_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg7_prop.load_npz(pc_data_folder_stg7 + "Proppant Concentration.npz")
pc_stg7_slurry_rate.load_npz(pc_data_folder_stg7 + "Slurry Rate.npz")
pc_stg7_pressure.load_npz(pc_data_folder_stg7 + "Treating Pressure.npz")

pc_data_folder_stg8 = datapath + "prod/pumping_data/stage8/"
pc_stg8_prop = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_slurry_rate = Data1D_PumpingCurve.Data1DPumpingCurve()
pc_stg8_pressure = Data1D_PumpingCurve.Data1DPumpingCurve()

pc_stg8_prop.load_npz(pc_data_folder_stg8 + "Proppant Concentration.npz")
pc_stg8_slurry_rate.load_npz(pc_data_folder_stg8 + "Slurry Rate.npz")
pc_stg8_pressure.load_npz(pc_data_folder_stg8 + "Treating Pressure.npz")

#%% Get the time range of stages
stg7_bgtime = pc_stg7_slurry_rate.get_start_time()
stg7_edtime = pc_stg7_slurry_rate.get_end_time()

stg8_bgtime = pc_stg8_slurry_rate.get_start_time()
stg8_edtime = pc_stg8_slurry_rate.get_end_time()

#%% Load DAS data
DASdata_stg7_path = datapath + "h_well/DAS/LFDASdata_stg7_hwell.npz"
DASdata_stg7interval_path = datapath + "h_well/DAS/LFDASdata_stg7_interval_hwell.npz"
DASdata_stg8_path= datapath + "h_well/DAS/LFDASdata_stg8_hwell.npz"

DASdata = Data2D_XT_DSS.DSS2D()
DASdata.load_npz(DASdata_stg7_path)

DASdata_tmp = Data2D_XT_DSS.DSS2D()
DASdata_tmp.load_npz(DASdata_stg7interval_path)

DASdata.right_merge(DASdata_tmp)

DASdata_tmp.load_npz(DASdata_stg8_path)
DASdata.right_merge(DASdata_tmp)

#%% Select the area of interest. Only crop depth
# Legacy dataformat

frac_hit_datapath = f"data/legacy/h_well/geometry/frac_hit/"
frac_hit_dataframe_stage7 = np.load(frac_hit_datapath + f"frac_hit_stage_{stage1}_hwell.npz")
frac_hit_dataframe_stage8 = np.load(frac_hit_datapath + f"frac_hit_stage_{stage2}_hwell.npz")

# extract data
frac_hit_stg7 = frac_hit_dataframe_stage7['data']
frac_hit_stg8 = frac_hit_dataframe_stage8['data']

DASdata.select_depth(np.min(frac_hit_stg8) - 500, np.max(frac_hit_stg7) + 500)

gauge_md_datapath = f"data/legacy/h_well/geometry/gauge_md_hwell.npz"
gauge_md_datapath = f"data/legacy/h_well/geometry/gauge_md_hwell.npz"
gauge_md = np.load(gauge_md_datapath)
gauge_md = gauge_md['data']
# filter out the gauge md
ind = np.array(np.where(np.logical_and(gauge_md <=  np.max(frac_hit_stg7) + 500, gauge_md >= np.min(frac_hit_stg8) - 500))).flatten()

gauge_dataframe_all = []
for iter in tqdm(ind):
    datapath = f'data/fiberis_format/h_well/gauges/gauge{iter+1}_data_hwell.npz'
    gauge_dataframe = Data1D_Gauge.Data1DGauge()
    gauge_dataframe.load_npz(datapath)
    gauge_dataframe.crop(stg7_bgtime, stg8_edtime)
    gauge_dataframe_all.append(gauge_dataframe)

#%% Setup for the plot
from datetime import timedelta
cx = np.array([-1, 1])

scalar_taxis = np.repeat(gauge_dataframe_all[0].start_time + timedelta(minutes=30), 2)
scalar_value = 500
scalar_tmp_value = np.array([+ gauge_md[ind][0] + 140, (scalar_value) * - coeff + gauge_md[ind][0] + 140])

#%% Concatenate the data to avoid labelling issue
pc_stg7_prop.right_merge(pc_stg8_prop)
pc_stg7_prop.rename("Proppant Concentration")

# pc_stg7_pressure.right_merge(pc_stg8_pressure)
# pc_stg7_pressure.rename("Treating Pressure")

pc_stg7_slurry_rate.right_merge(pc_stg8_slurry_rate)
pc_stg7_slurry_rate.rename("Slurry Rate")

#%% create figure
# fig, ax1 = plt.subplots()
#
# for i in range(len(gauge_md[ind])):
#     ax1.axhline(y=gauge_md[ind][i], color='black', linestyle='--')
#     datetime_taxis = gauge_dataframe_all[i].calculate_time()
#     ax1.plot(datetime_taxis, (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * - coeff + gauge_md[ind][i], color='cyan', linewidth=2)
# ax1.plot(scalar_taxis, scalar_tmp_value, color='black', linewidth=5)
# ax1.text(scalar_taxis[0] + timedelta(minutes=8), scalar_tmp_value[0] - scalar_value/7, f"{scalar_value} psi", fontsize=12, color='black')
#
# img1 = DASdata.plot(ax=ax1, useTimeStamp=True, cmap='bwr', aspect='auto')
# img1.set_clim(cx * 3e2)
#
# plt.title("LF-DAS data with gauge in H well/stage 7 and 8")
# plt.tight_layout()
# plt.savefig("figs/02112025/output.png")
# plt.show()

# Next step is to add the pumping data into the plot.

#%% Plot with pumping data
plt.figure(figsize = (8, 5))

ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=4, rowspan=4)
for i in range(len(gauge_md[ind])):
    ax1.axhline(y=gauge_md[ind][i], color='black', linestyle='--')
    datetime_taxis = gauge_dataframe_all[i].calculate_time()
    ax1.plot(datetime_taxis, (gauge_dataframe_all[i].data - gauge_dataframe_all[i].data[0]) * - coeff + gauge_md[ind][i], color='cyan', linewidth=2)
ax1.plot(scalar_taxis, scalar_tmp_value, color='black', linewidth=5)
ax1.text(scalar_taxis[0] + timedelta(minutes=8), scalar_tmp_value[0] - scalar_value/7, f"{scalar_value} psi", fontsize=12, color='black')

img1 = DASdata.plot(ax=ax1, useTimeStamp=True, cmap='bwr', aspect='auto')
img1.set_clim(cx * 3e2)

ax2 = plt.subplot2grid((6, 4), (4, 0), colspan=4, rowspan=2, sharex = ax1)

color = 'blue'
pc_stg7_prop.plot(ax=ax2, useTimeStamp=True, title=None, color=color)
ax2.set_ylabel(fr"Prop. Conc./lb$\cdot$gal$^{-1}$", color=color)
ax2.tick_params(axis='y', labelcolor=color)
# set xlim
ax2.set_xlim(stg7_bgtime, stg8_edtime)

ax21 = ax2.twinx()
color = 'green'
pc_stg7_slurry_rate.plot(ax=ax21, useTimeStamp=True, title=None, color=color)
ax21.set_ylabel("Slurry Rate/bpm", color=color)
ax21.tick_params(axis='y', labelcolor=color)

ax22 = ax2.twinx()
color = 'red'
ax22.spines["right"].set_position(("outward", 60))
pc_stg7_pressure.plot(ax=ax22, useTimeStamp=True, title=None, color=color)
ax22.set_ylabel("Treating Pressure/psi", color=color)
ax22.tick_params(axis='y', labelcolor=color)

ax23 = ax2.twinx()
pc_stg8_pressure.plot(ax=ax23, useTimeStamp=True, title=None, color=color)
# Remove yaxis
ax23.yaxis.set_visible(False)

plt.suptitle("LF-DAS data with gauge in H well/stage 7 and 8")
plt.tight_layout()
plt.savefig("figs/02112025/well_coplot_figure/Hwell.png")
plt.show()