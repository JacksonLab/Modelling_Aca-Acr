import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--parameter",required=True)
parser.add_argument("--outdir",required=True)
parser.add_argument("--nsamples", type=int, default=1)
parser.add_argument("--threads", type=int, default=1)
args=parser.parse_args()
print ("Run modelling for parameter", args.parameter, "outdir is:", args.outdir)

import os, sys, subprocess

os.environ["NUMEXPR_MAX_THREADS"]="200"

import multiprocessing
import tqdm
import numpy as np
import scipy.stats as st
import numba
import math
import pandas as pd
from scipy.signal import argrelextrema
import scipy.fftpack
import os.path
import biocircuits

# Plotting modules
import bokeh.io
import bokeh.plotting
from bokeh.models import LinearColorMapper, ColorBar
from bokeh.io import export_svgs, export_png
from bokeh.models import Range1d, DataRange1d

from bokeh.io import output_notebook
from bokeh.core.validation.warnings import MISSING_RENDERERS
bokeh.core.validation.silence(MISSING_RENDERERS, True)

exportDir=args.outdir
runParameter=args.parameter

# Events and propensities matricies
simple_update = np.array(
  [ # u,  ux, m,  mx, x,  y
    [ 1,  0,  0,  0,  0,  0], # Unbound DNA replication
    [ 2, -1,  0,  0,  1,  0], # Bound DNA replication and displacement of Aca2
    [-1,  1,  0,  0, -1,  0], # Aca2 binding DNA
    [ 1, -1,  0,  0,  1,  0], # Aca2 dissociating from DNA
    [ 0,  0,  1,  0,  0,  0], # Transcription of unbound DNA
    [ 0,  0, -1,  1, -1,  0], # Aca2 binding RNA
    [ 0,  0,  1, -1,  1,  0], # Aca2 dissociating from RNA
    [ 0,  0, -1,  0,  0,  0], # free RNA decay
    [ 0,  0,  0, -1,  1,  0], # Bound RNA decay and release of Aca2
    [ 0,  0,  0,  0,  1,  0], # Aca2 translation from free RNA
    [ 0,  0,  0,  0,  1,  0], # Aca2 translation from bound RNA
    [ 0,  0,  0,  0, -1,  0], # Aca2 decay
    [ 0,  0,  0,  0,  0,  1], # Acr translation from free RNA
    [ 0,  0,  0,  0,  0, -1], # Acr decay
    ],
    dtype=int,
 )

def simple_propensity(propensities, state, t, alpha, beta, theta, phi, delta, am, dm, au, du, umax, r, latency):
    u, ux, m, mx, x, y = state
    propensities[0]  = ((t - latency) > 0) * r*(1 - (u + ux)/umax)*u   # Unbound DNA replication
    propensities[1]  = ((t - latency) > 0) * r*(1 - (u + ux)/umax)*ux  # Bound DNA replication and displacement of Aca2
    propensities[2]  = au*u*x                    # Aca2 binding DNA
    propensities[3]  = du*ux                     # Aca2 dissociating from DNA
    propensities[4]  = alpha*u                   # Transcription of unbound DNA
    propensities[5]  = am*m*x                    # Aca2 binding RNA
    propensities[6]  = dm*mx                     # Aca2 dissociating from RNA
    propensities[7]  = phi*m                     # Free RNA decay
    propensities[8]  = phi*mx                    # Bound RNA decay and release of Aca2
    propensities[9]  = beta*m                    # Aca2 translation from free RNA
    propensities[10] = beta*mx                   # Aca2 translation from bound RNA
    propensities[11] = delta*x                   # Aca2 decay
    propensities[12] = theta*m                   # Acr translation from free RNA
    propensities[13] = delta*y                   # Acr decay

# parameters to deactivate for different regulation scenarios

#                alpha, beta, theta, phi, delta, am, dm, au, du, utk, r, latency)
sampleConfig = [(1,     1,    1,     1,   1,     0,  1,  0,  1,  1,   1, 1), # no regulation
                (1,     1,    1,     1,   1,     1,  1,  0,  1,  1,   1, 1), # RNA-only
                (1,     1,    1,     1,   1,     0,  1,  1,  1,  1,   1, 1), # DNA-only
                (1,     1,    1,     1,   1,     1,  1,  1,  1,  1,   1, 1)] # full reg

#print(sampleConfig)

# Parameter settings

timescaleFactor = 60  # convert seconds to min

### parameters in human-friendly formats (used for plot axes labels) ###

# first-order mesoscopic
phi0 = 10         # units = min        # mRNA halflife (converted later to rate)

# second-order mesoscopic (1/(molecules x seconds))
alpha0 = 5       # units = 1/min/DNA   # Transcription rate
beta0 = 2        # units = 1/min/mRNA  # Translation rate for Aca2 (dimers)
theta0 = 20      # units = 1/min/mRNA  # Translation rate of Acr
delta = 0        # unused protein turnover

# second-order macroscopic (1/(molar x seconds))
# > assume 1 molecule per cell = 1 nM > second-order mesoscopic > pseudo first-order

# experimental estimates
kdu0 = 1.4        # units = nM          # Aca2-DNA dissociation constant (Kd)
kdm0 = 30.2       # units = nM          # Aca2-RNA dissociation constant (Kd)

# approximations for K.on
au0 = 0.01       # units = 1/(nM.s) >> 1/(molecule.s) (1 nM = 1 molecule/cell)  # Aca2-DNA association rate (K.on)
am0 = 0.01       # units = 1/(nM.s) >> 1/(molecule.s) (1 nM = 1 molecule/cell)  # Aca2-RNA association rate (K.on)

# miscellaneous
moi = 1          # units = molecules   # Number of DNA molecules (unbound) at t=0
latency = 5      # units = min         # Latent period before DNA replication begins
umax = 50        # units = molecules   # Total DNA (maximum)
r = 0.2          # units = 1/min       # DNA replication

### parameter sweeping ###

paramsize = 5    # maximum number of elements for paramater sweeps

umax_v = (1, 10, 50, 100, 200)          # Varying total number of DNA
phi_v0 = (2, 5, 10, 15, 30)             # Varying mRNA decay rate (min)
beta_v0 = (0.5, 1, 2, 5, 10)            # Varying translation rate (Aca2 dimers)

moi_v = (1, 2, 5, 10, 20)               # Varying initial number of DNA molecules at t=0
r_v = (0.05, 0.1, 0.2, 0.5, 1)       # Varying DNA replication
alpha_v0 = (1, 2, 5, 10, 20)        # Varying transcription rate
theta_v0 = (5, 10, 20, 50, 100)         # Varying translation rate (Acr) # only changes the output gain

kdu_v0 = (0.1, 0.5, 1.4, 5, 50)      # Varying dissociation constant (Aca2 to DNA)
kdm_v0 = (1, 10, 30.2, 100, 250)      # Varying dissociation constant (Aca2 to RNA)

au_v0 = (0.001, 0.005, 0.01, 0.05, 0.1)           # Varying association rate (Aca2 to DNA) K.on
am_v0 = (0.001, 0.005, 0.01, 0.05, 0.1)           # Varying association rate (Aca2 to RNA) K.on

### convert parameters to required format and scale ###

# convert RNA decay rate from t 1/2 to constant
phi = np.log(2)/phi0
phi_v = (tuple([ np.log(2)/val for val in phi_v0]))

# adjust any other required parameters to the correct timescale or units
am = am0 * timescaleFactor # 1/sec to 1/min
au = au0 * timescaleFactor # 1/sec to 1/min

am_v = (tuple([val*timescaleFactor for val in am_v0]))  # 1/sec to 1/min
au_v = (tuple([val*timescaleFactor for val in au_v0]))  # 1/sec to 1/min

# convert Kd to K.off using mesoscopic K.on in 1/(molecules.min)
du = kdu0*au           # units = 1/min       # Aca2-DNA dissociation rate (K.off)
dm = kdm0*am           # units = 1/min       # Aca2-RNA dissociation rate (K.off)

du_v = (tuple([val*au for val in kdu_v0]))
dm_v = (tuple([val*am for val in kdm_v0]))

# unchanged units
alpha = alpha0
beta = beta0
theta = theta0

alpha_v = alpha_v0
beta_v = beta_v0
theta_v = theta_v0

### runtime settings

N = args.nsamples          # Sample size per thread
n = args.threads           # Threads
idAca = 4       # index of Aca counts in output array
idAcr = 5       # index of Acr counts in output array

time_points1 = np.linspace(0, 60, 121) # Reporting intervals
# initial state
population_0 = np.zeros(6, dtype=int)
population_0[0] = moi

f = 1 # times std for shaded region

# unused: for selective running of only some samples
numSamples=4
samplesToRun=(0,1,2,3)

###

# Aesthetics
palette1 = ('#fdd49e','#fc8d59','#b30000')
palette3 = ('black','blue')

paletteSweepAcr = ('#fdd49e','#fc8d59','#fc8d59','#f26066','#b30000') # lightest -> darkest
paletteSweepAca = ('#77ccff','#3388ff','#3388ff','#3388ff','#0044ff')
paletteSweepDNA = ('#94c58c','#429b46','#429b46','#429b46','#0a6921')
paletteSweepRNA = ('#e1affd','#ca8dfd','#ca8dfd','#ca8dfd','#b042ff')

paletteEndpointComp = ('black','#377eb8','#4daf4a','#e41a1c')

linestyle = ["solid","dashed"]

# Set the number of plot types (columns) for the parameter sweeps
plotCount = 6

label = ['AcrF8','Free Aca2']
title = ['Unregulated','RNA regulation only','DNA regulation only','DNA/RNA regulation','+/- RNAreg. Dash = +RNAreg']

# Set up plots
fig_size = [300, 300]
x_range = (0, time_points1[-1])


# Plotting and data export function

def GetData(exportDF, data, rate, paramName, sampleNum, dataLabel):
    m = np.mean(data, axis = 0)
    sig = np.std(data, axis = 0)

    sampleNames = ["none","RNA","DNA","both"]
    # if the data is a rate, stack 0 at t=0 to ensure the array lengths are the same as the timepoints
    if rate == True:
        m = np.hstack((0,m))
        sig = np.hstack((0,sig))

    # convert to DF and label columns
    m = pd.DataFrame(m,columns=[dataLabel + '.mean.'  + str(paramName) + '.' + sampleNames[sampleNum]])
    sig = pd.DataFrame(sig,columns=[dataLabel + '.sd.' + str(paramName) + '.' + sampleNames[sampleNum]])
    exportDF = pd.concat([exportDF.reset_index(drop = True), m,sig], axis = 1)

    return exportDF

def GeneratePlots(param_v,samples1,samples2,samples3,samples4,plotCount,parameterSwept):
  plots = []
  data_io = []
  plotTypes=11
  y_range = (0, 1500)
  for i in range(plotCount*plotTypes):
    plots.append(bokeh.plotting.figure(width=fig_size[0], height=fig_size[1],x_range=x_range, y_range=y_range),)
    plots[i].axis.major_label_text_font_size = "10pt"
    plots[i].axis.axis_label_text_font_size = "10pt"
    plots[i].axis.axis_label_text_font_style = "normal"

  # for kinetic plots
  outputs = np.empty((4,plotTypes,N*n,paramsize))

  # data frame for export
  exportDF = pd.DataFrame(time_points1, columns=["min"]) # define dataframe for export

  # only plot the first, middle and end datasets (for clarity)
  toPlot = [0,2,4]

  for i, paramValue in enumerate(param_v):
    data = [samples1[i], samples2[i], samples3[i], samples4[i]]
    for k, samples in enumerate(data):
        acr = samples[:,:,idAcr]
        acrRate = acr[:,1::] - acr[:,0:-1:]
        m = np.mean(acr, axis = 0)
        sig = np.std(acr, axis = 0)
        if i in toPlot:
          plots[k].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepAcr[i], alpha = 0.1)
          plots[k].line(time_points1, np.mean(acr, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAcr[i])
          plots[k + plotCount*1].line(time_points1[0:-1:], np.mean(acrRate, axis=0)/(time_points1[2]-time_points1[1]), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAcr[i])
        # calculate max rate (of the mean) for kinetics
        timestep=(time_points1[2]-time_points1[1])
        mRate = (m[1::] - m[0:-1:])/timestep
        outputs[k,1][:,i] = np.max(mRate)

        aca = samples[:,:,idAca] + samples[:,:,1] + samples[:,:,3] # Total Aca2: 1=bound DNA, 3=bound RNA, 4=Free Aca
        acaRate = aca[:,1::] - aca[:,0:-1:]
        m = np.mean(aca, axis = 0)
        sig = np.std(aca, axis = 0)
        if i in toPlot:
          plots[k + plotCount*2].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepAca[i], alpha = 0.1)
          plots[k + plotCount*2].line(time_points1, np.mean(aca, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i])
          plots[k + plotCount*3].line(time_points1[0:-1:], np.mean(acaRate, axis=0)/(time_points1[2]-time_points1[1]), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i])
        # calculate max rate (of the mean) for kinetics
        timestep=(time_points1[2]-time_points1[1])
        mRate = (m[1::] - m[0:-1:])/timestep
        outputs[k,3][:,i] = np.max(mRate)

        acaFree = samples[:,:,idAca] # Free Aca2
        acaFreeRate = acaFree[:,1::] - acaFree[:,0:-1:]
        m = np.mean(acaFree, axis = 0)
        sig = np.std(acaFree, axis = 0)
        if i in toPlot:
          plots[k + plotCount*4].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepAca[i], alpha = 0.1)
          plots[k + plotCount*4].line(time_points1, np.mean(acaFree, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i])
          plots[k + plotCount*5].line(time_points1[0:-1:], np.mean(acaFreeRate, axis=0)/(time_points1[2]-time_points1[1]), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i])
        # calculate max rate (of the mean) for kinetics
        timestep=(time_points1[2]-time_points1[1])
        mRate = (m[1::] - m[0:-1:])/timestep
        outputs[k,5][:,i] = np.max(mRate)

        dnaTotal = samples[:,:,0] + samples[:,:,1]  # Total DNA/promoters
        m = np.mean(dnaTotal, axis = 0)
        sig = np.std(dnaTotal, axis = 0)
        if i in toPlot:
          plots[k + plotCount*6].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepDNA[i], alpha = 0.1)
          plots[k + plotCount*6].line(time_points1, np.mean(dnaTotal, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepDNA[i])

        dnaFree = samples[:,:,0] # Free DNA/promoters
        m = np.mean(dnaFree, axis = 0)
        sig = np.std(dnaFree, axis = 0)
        if i in toPlot:
          plots[k + plotCount*7].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepDNA[i], alpha = 0.1)
          plots[k + plotCount*7].line(time_points1, np.mean(dnaFree, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepDNA[i])

        rnaTotal = samples[:,:,2] + samples[:,:,3]  # Total RNA
        m = np.mean(rnaTotal, axis = 0)
        sig = np.std(rnaTotal, axis = 0)
        if i in toPlot:
          plots[k + plotCount*8].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepRNA[i], alpha = 0.1)
          plots[k + plotCount*8].line(time_points1, np.mean(rnaTotal, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepRNA[i])

        rnaFree = samples[:,:,2] # Free RNA
        m = np.mean(rnaFree, axis = 0)
        sig = np.std(rnaFree, axis = 0)
        if i in toPlot:
          plots[k + plotCount*9].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepRNA[i], alpha = 0.1)
          plots[k + plotCount*9].line(time_points1, np.mean(rnaFree, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepRNA[i])

        rnaBound = samples[:,:,3] # Bound RNA
        m = np.mean(rnaBound, axis = 0)
        sig = np.std(rnaBound, axis = 0)
        if i in toPlot:
          plots[k + plotCount*10].varea(time_points1, m - f*sig, m + f*sig ,color=paletteSweepRNA[i], alpha = 0.1)
          plots[k + plotCount*10].line(time_points1, np.mean(rnaBound, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepRNA[i])

        # export data to csv (mean +/- std.dev)
        exportDF = GetData(exportDF, acr, False,paramValue,k,"#acr")
        exportDF = GetData(exportDF, acrRate, True,paramValue,k,"acr/min")
        exportDF = GetData(exportDF, aca, False,paramValue,k,"#aca")
        exportDF = GetData(exportDF, acaRate, True,paramValue,k,"aca/min")
        exportDF = GetData(exportDF, acaFree, False,paramValue,k,"#aca.free")
        exportDF = GetData(exportDF, acaFreeRate, True,paramValue,k,"aca.free/min")
        exportDF = GetData(exportDF, dnaTotal, False,paramValue,k,"#DNA")
        exportDF = GetData(exportDF, dnaFree, False,paramValue,k,"#DNA.free")
        exportDF = GetData(exportDF, rnaTotal, False,paramValue,k,"#RNA")
        exportDF = GetData(exportDF, rnaFree, False,paramValue,k,"#RNA.free")
        exportDF = GetData(exportDF, rnaBound, False,paramValue,k,"#RNA.bound")

    # save the dataframe as a csv file
    #DF = pd.DataFrame(exportDF.transpose())
    exportDF.to_csv(exportDir + 'aca2_modelling_output_' + parameterSwept + '.csv')

    # store data for kinetic plots

    # Acr
    outputs[0,0][:,i] = samples1[i][:,-1,idAcr]
    outputs[1,0][:,i] = samples2[i][:,-1,idAcr]
    outputs[2,0][:,i] = samples3[i][:,-1,idAcr]
    outputs[3,0][:,i] = samples4[i][:,-1,idAcr]

    # Total Aca
    outputs[0,2][:,i] = samples1[i][:,-1,idAca] + samples1[i][:,-1,1] + samples1[i][:,-1,3]
    outputs[1,2][:,i] = samples2[i][:,-1,idAca] + samples2[i][:,-1,1] + samples2[i][:,-1,3]
    outputs[2,2][:,i] = samples3[i][:,-1,idAca] + samples3[i][:,-1,1] + samples3[i][:,-1,3]
    outputs[3,2][:,i] = samples4[i][:,-1,idAca] + samples4[i][:,-1,1] + samples4[i][:,-1,3]

    # Free Aca
    outputs[0,4][:,i] = samples1[i][:,-1,idAca]
    outputs[1,4][:,i] = samples2[i][:,-1,idAca]
    outputs[2,4][:,i] = samples3[i][:,-1,idAca]
    outputs[3,4][:,i] = samples4[i][:,-1,idAca]

    # make plots for +/- RNA regulation comparison
    if i in toPlot:
      data = [samples3[i], samples4[i]]
      for k, samples in enumerate(data):
        acr = samples[:,:,idAcr] # samples[:,:,10][0]
        m = np.mean(acr, axis = 0)
        sig = np.std(acr, axis = 0)
        plots[plotCount*0 + 4].line(time_points1, np.mean(acr, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAcr[i],line_dash=linestyle[k])
        acrRate = acr[:,1::] - acr[:,0:-1:]
        plots[plotCount*1 + 4].line(time_points1[0:-1:], np.mean(acrRate, axis=0)/(time_points1[2]-time_points1[1]), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAcr[i],line_dash=linestyle[k])

        aca = samples[:,:,idAca] + samples[:,:,1] + samples[:,:,3] # Total Aca2: 1=bound DNA, 3=bound RNA, 4=Free Aca
        m = np.mean(aca, axis = 0)
        sig = np.std(aca, axis = 0)
        plots[plotCount*2 + 4].line(time_points1, np.mean(aca, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i],line_dash=linestyle[k])
        acaRate = aca[:,1::] - aca[:,0:-1:]
        plots[plotCount*3 + 4].line(time_points1[0:-1:], np.mean(acaRate, axis=0)/(time_points1[2]-time_points1[1]), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i],line_dash=linestyle[k])

        acaFree = samples[:,:,idAca] # Free Aca2
        m = np.mean(acaFree, axis = 0)
        sig = np.std(acaFree, axis = 0)
        plots[plotCount*4 + 4].line(time_points1, np.mean(acaFree, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i],line_dash=linestyle[k])
        acaFreeRate = acaFree[:,1::] - acaFree[:,0:-1:]
        plots[plotCount*5 + 4].line(time_points1[0:-1:], np.mean(acaFreeRate, axis=0)/(time_points1[2]-time_points1[1]), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepAca[i],line_dash=linestyle[k])

        dnaTotal = samples[:,:,0] + samples[:,:,1]  # Total DNA/promoters
        m = np.mean(dnaTotal, axis = 0)
        sig = np.std(dnaTotal, axis = 0)
        plots[plotCount*6 + 4].line(time_points1, np.mean(dnaTotal, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepDNA[i],line_dash=linestyle[k])

        dnaFree = samples[:,:,0] # Free DNA/promoters
        m = np.mean(dnaFree, axis = 0)
        sig = np.std(dnaFree, axis = 0)
        plots[plotCount*7 + 4].line(time_points1, np.mean(dnaFree, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepDNA[i],line_dash=linestyle[k])

        rnaTotal = samples[:,:,2] + samples[:,:,3]  # Total RNA
        m = np.mean(rnaTotal, axis = 0)
        sig = np.std(rnaTotal, axis = 0)
        plots[plotCount*8 + 4].line(time_points1, np.mean(rnaTotal, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepRNA[i],line_dash=linestyle[k])

        rnaFree = samples[:,:,2] # Free RNA
        m = np.mean(rnaFree, axis = 0)
        sig = np.std(rnaFree, axis = 0)
        plots[plotCount*9 + 4].line(time_points1, np.mean(rnaFree, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepRNA[i],line_dash=linestyle[k])

        rnaBound = samples[:,:,3] # Bound RNA
        m = np.mean(rnaBound, axis = 0)
        sig = np.std(rnaBound, axis = 0)
        plots[plotCount*10 + 4].line(time_points1, np.mean(rnaBound, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteSweepRNA[i],line_dash=linestyle[k])

  for i in range(5):
    for p in range(plotTypes):
      plots[i + plotCount*p].xaxis.axis_label = 'Time (min)'
      plots[i + plotCount*p].title = title[i]


  for i in range(plotCount):

    plots[i + plotCount*0].yaxis.axis_label  = 'Acr (# molecules)'
    plots[i + plotCount*1].yaxis.axis_label  = 'Acr rate (# molecules/min)'
    plots[i + plotCount*2].yaxis.axis_label  = 'Total Aca2 (# molecules)'
    plots[i + plotCount*3].yaxis.axis_label  = 'Total Aca2 rate (# molecules/min)'
    plots[i + plotCount*4].yaxis.axis_label  = 'Free Aca2 (# molecules)'
    plots[i + plotCount*5].yaxis.axis_label  = 'Free Aca2 rate (# molecules/min)'
    plots[i + plotCount*6].yaxis.axis_label  = 'Total DNA (# molecules)'
    plots[i + plotCount*7].yaxis.axis_label  = 'Unbound DNA (# molecules)'
    plots[i + plotCount*8].yaxis.axis_label  = 'Total RNA (# molecules)'
    plots[i + plotCount*9].yaxis.axis_label  = 'Free RNA (# molecules)'
    plots[i + plotCount*10].yaxis.axis_label = 'Bound RNA (# molecules)'

  # Kinetic plots

  for p in range(6):
    data = [outputs[0,p], outputs[1,p], outputs[2,p], outputs[3,p]]
    for i, y in enumerate(data):
      m = np.mean(y, axis = 0)
      sig = np.std(y, axis = 0)
      plots[plotCount*(p+1)-1].varea(param_v, m - f*sig, m + f*sig ,color=paletteEndpointComp[i], alpha = 0.1)
      plots[plotCount*(p+1)-1].line(param_v, np.mean(y, axis=0), line_width=2, alpha= 1, line_join="bevel",color=paletteEndpointComp[i])
      plots[plotCount*(p+1)-1].circle(param_v, np.mean(y, axis=0),size=8,color=paletteEndpointComp[i])
      plots[plotCount*(p+1)-1].y_range = DataRange1d()
      plots[plotCount*(p+1)-1].x_range = DataRange1d()
      plots[plotCount*(p+1)-1].xaxis.axis_label = parameterSwept


  plots[5].title =  'Endpoint (60 min) red = fullReg'
  plots[11].title = 'Max rate red=fullReg'
  plots[17].title = 'Endpoint (60 min) red = fullReg'
  plots[23].title = 'Max rate red=fullReg'
  plots[29].title = 'Endpoint (60 min) red = fullReg'
  plots[35].title = 'Max rate red=fullReg'

  for i,fig in enumerate(plots):
    if not fig.renderers:
        fig.visible = False

  return (plots)

#Sweep: Maximum total DNA / copy number (starts at one genome then replicates to reach the specified level):
if runParameter == "copy_number":

   param_v = umax_v
   parameterSwept = 'DNA capacity'

   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)

   for i, umaxk in enumerate(param_v):
       argsBase = (alpha, beta, theta, phi, delta, am, dm, au, du, umaxk, r, latency)
       args= [argsBase] * 4

       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)

   plots = GeneratePlots(param_v,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept)

# set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 2500)                  # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 200)    # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 500)     # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 50)    # Total Aca rate
       plots[i + plotCount*4].y_range = Range1d(0, 1000)     # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 100)    # Free Aca rate
       plots[i + plotCount*6].y_range = Range1d(0, 210)     # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)     # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 200)     # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 10)     # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 10)    # Bound RNA

   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png')

# Sweep: Replication rate
elif runParameter == "replication_rate":
   param_v = r_v
   parameterSwept = 'Replication rate'

   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)

   for i, rk in enumerate(param_v):
       argsBase = (alpha, beta, theta, phi, delta, am, dm, au, du, umax, rk, latency)
       args= [argsBase] * 4

       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   plots = GeneratePlots(param_v,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept)

   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 10000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 600)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 2000)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 500)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 300)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 500)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 150)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 150)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 150)    # Bound RNA

   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png')

#Sweep: Transcription rate
elif runParameter == "transcription_rate":
   param_v = alpha_v
   parameterSwept = 'Transcription rate'
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)

   for i, alphak in enumerate(param_v):
       argsBase = (alphak, beta, theta, phi, delta, am, dm, au, du, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(param_v,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept)
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 8000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 500)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 5000)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 500)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 5000)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 500)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)     # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, umax+10)     # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 300)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 300)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 300)    # Bound RNA
 
   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png') 

#Sweep: Aca2 translation rate
elif runParameter == "translation_rate":   
   param_v = beta_v
   parameterSwept = 'Aca translation rate'
   
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)
   
   for i, betak in enumerate(param_v):
       argsBase = (alpha, betak, theta, phi, delta, am, dm, au, du, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(param_v,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept)
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 10000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 200)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 5000)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 200)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 500)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 5000)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 50)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 50)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 50)    # Bound RNA

   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png')   

#@title Sweep: Aca2 mRNA association rate
elif runParameter == "rna_kon":
   param_v = am_v
   parameterSwept = 'Aca2-RNA k.on'
   
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)
   
   for i, amk in enumerate(param_v):
       # update k.off (dm) (assume the Kd stays the same as experimentally determined)
       dmk=dm*(am/amk)
   
       argsBase = (alpha, beta, theta, phi, delta, amk, dmk, au, du, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(am_v0,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept) # pass human-friendly elements for swept parameter
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 8000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 300)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 1000)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 100)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 1000)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 100)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 50)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 50)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 50)    # Bound RNA
 
   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png') 
  
#Sweep: Aca2 DNA association rate
elif runParameter == "dna_kon":
   param_v = au_v
   parameterSwept = 'Aca-DNA k.on'
   
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)
   
   for i, auk in enumerate(param_v):
       # update k.off (dm) (assume the Kd stays the same as experimentally determined)
       duk=du*(au/auk)
   
       argsBase = (alpha, beta, theta, phi, delta, am, dm, auk, duk, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(au_v0,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept) # pass human-friendly elements for swept parameter
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 5000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 500)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 1000)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 100)   # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 1000)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 500)   # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 50)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 50)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0,50)    # Bound RNA
 
   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png') 

#Sweep: RNA decay rate
elif runParameter == "rna_decay":
   param_v = phi_v
   parameterSwept = 'RNA decay'
   
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)
   
   for i, phik in enumerate(param_v):
       argsBase = (alpha, beta, theta, phik, delta, am, dm, au, du, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(phi_v0,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept) # pass human-friendly elements for swept parameter
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 10000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 300)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 1500)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 100)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 1500)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 100)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 150)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 150)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 150)    # Bound RNA
 
   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png') 
  
#Sweep: Aca2-RNA dissociation constant
elif runParameter == "rna_kd":
   param_v = dm_v
   parameterSwept = 'Aca2-RNA Kd (run as k.off)'
   
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)
   
   for i, dmk in enumerate(param_v):
       # update K.on (am) based on Kd relative to k.off
       amk=am*(dm/dmk)
   
       argsBase = (alpha, beta, theta, phi, delta, amk, dmk, au, du, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(kdm_v0,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept) # pass human-friendly elements for swept parameter - use Kd not K.off
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 10000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 300)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 1500)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 100)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 500)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 500)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 50)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 50)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0,50)    # Bound RNA

   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png')
   
#Sweep: Aca2-DNA dissociation constant
elif runParameter == "dna_kd":
   param_v = du_v
   parameterSwept = 'Aca2-DNA Kd (run as k.off)'
   
   samples = np.empty((numSamples,paramsize, N*n, len(time_points1), 6), dtype=int)
   
   for i, duk in enumerate(param_v):
       # update k.on (am) based on Kd relative to k.off
       auk=au*(du/duk)
   
       argsBase = (alpha, beta, theta, phi, delta, am, dm, auk, duk, umax, r, latency)
       args= [argsBase] * 4
   
       for run in samplesToRun:
         args[run] = [argsBase[element] * sampleConfig[run][element] for element in range(12)]
         samples[run,i] = biocircuits.gillespie_ssa(simple_propensity, simple_update, population_0,time_points1, size=N, args=tuple(args[run]), n_threads=n,progress_bar=True)
   
   plots = GeneratePlots(kdu_v0,samples[0],samples[1],samples[2],samples[3],plotCount,parameterSwept) # pass human-friendly elements for swept parameter - use Kd not K.off
   
   # set y axis limits
   for i in range(plotCount):
       plots[i].y_range = Range1d(0, 25000)                 # Acr
       plots[i + plotCount*1].y_range = Range1d(0, 500)   # Acr rate
       plots[i + plotCount*2].y_range = Range1d(0, 1500)    # Total Aca
       plots[i + plotCount*3].y_range = Range1d(0, 300)    # Total Aca2 rate
       plots[i + plotCount*4].y_range = Range1d(0, 300)    # Free Aca
       plots[i + plotCount*5].y_range = Range1d(0, 300)    # Free Aca2 rate
       plots[i + plotCount*6].y_range = Range1d(0, umax+10)    # Total DNA
       plots[i + plotCount*7].y_range = Range1d(0, 10)    # Unbound DNA
       plots[i + plotCount*8].y_range = Range1d(0, 50)    # Total RNA
       plots[i + plotCount*9].y_range = Range1d(0, 50)    # Unbound RNA
       plots[i + plotCount*10].y_range = Range1d(0, 50)    # Bound RNA

   grid = bokeh.layouts.gridplot(plots,ncols=plotCount)
   export_png(grid, filename=exportDir + runParameter + '.png') 

else:
   print("parameter not found")

print("complete, exiting")

   
   
   