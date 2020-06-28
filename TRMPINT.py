#import turbosensei as fs
from numba import jit
import numpy as np
from scipy.linalg import lstsq
from matplotlib.figure import figaspect
from math import acos, pi, degrees, sin, cos, sqrt, log, log10, tanh, remainder
from scipy.interpolate import interp2d
import random
import codecs as cd
import numpy as np

import ipywidgets as widgets
from ipywidgets import VBox, HBox
import codecs as cd
import numpy as np
import matplotlib.pyplot as plt


from scipy.linalg import lstsq

from numpy.polynomial.polynomial import polyfit

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib as matplotlib

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.interpolate import griddata

from ipyfilechooser import FileChooser

mu0 = 4*pi*1e-7
ms0 = 491880.5
kb = 1.3806503e-23
tau = 10e-9
roottwohffield = 2**(0.5)  

sumegli = 0.0
eglip=0.54
eglid=log(113*((31486**(-eglip))))+0.52
eglid=113*((31486**(-eglip)))
egliin = 5.4
eglip=egliin/10.0 # 0.54 #eglin from data file
hf=(10000**(0.54))*(10**(-0.52)) #checking maths
eglid=+0.52+(log10(hf)-log10(10000.0)*eglip)
affield = 0
affmax = 0.0
af_step_max = 0.0
flatsub=0
flat=0


afone = 1
afzero = 0

#FORC file read 

def FORCfile():
    fc = FileChooser()
    display(fc)
    return fc


#### file parsing routines

def parse_header(file,string):
    """Function to extract instrument settings from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    string: instrument setting to be extracted (string)
    Outputs:
    output: value of instrument setting [-1 if setting doesn't exist] (float)
    """
    output=-1 #default output (-1 corresponds to no result, i.e. setting doesn't exist)
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=float(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=float(line[idx+1:])  #value taken as everything to right of the setting name 

    return output


def parse_measurements(file):
    """Function to extract measurement points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    Ft: Estimated times at which the points were measured (float, seconds)
    dH: Measurement field spacing [float SI units]
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## determine indicies of each FORC
    idxSAT = np.array(np.where(np.isin(H0, dum))) #find start address of each blank line
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSTART = idxSAT[1::2]+1 #find start address of each FORC
    idxEND = idxSAT[2::2]-1 ##find end address of each FORC

    
    #Extract first FORC to initialize arrays 
    M=M0[idxSTART[0]:idxEND[0]+1] #Magnetization values
    H=H0[idxSTART[0]:idxEND[0]+1] #Field values
    Hr=np.ones(idxEND[0]+1-idxSTART[0])*H0[idxSTART[0]] #Reversal field values
    Fk=np.ones(idxEND[0]+1-idxSTART[0]) #index number of FORC
    Fj=np.arange(1,1+idxEND[0]+1-idxSTART[0])# measurement index within given FORC

    #Extract remaining FORCs one by one into into a long-vector
    for i in range(1,idxSTART.size):
        M=np.concatenate((M,M0[idxSTART[i]:idxEND[i]+1]))
        H=np.concatenate((H,H0[idxSTART[i]:idxEND[i]+1]))
        Hr=np.concatenate((Hr,np.ones(idxEND[i]+1-idxSTART[i])*H0[idxSTART[i]]))
        Fk=np.concatenate((Fk,np.ones(idxEND[i]+1-idxSTART[i])+i))
        Fj=np.concatenate((Fj,np.arange(1,1+idxEND[i]+1-idxSTART[i])))
    
    unit = parse_units(file) #Ensure use of SI units
    
    if unit=='Cgs':
        H=H/1E4 #Convert Oe into T
        Hr=Hr/1E4 #Convert Oe into T
        M=M/1E3 #Convert emu to Am^2

    dH = np.mean(np.diff(H[Fk==np.max(Fk)])) #mean field spacing

    Ft=measurement_times(file,Fk,Fj) #estimated time of each measurement point

    return H, Hr, M, Fk, Fj, Ft, dH
def parse_units(file):
    """Function to extract instrument unit settings ('') from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    CGS [Cgs setting] or SI [Hybrid SI] (string)
    """
    string = 'Units of measure' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idxSI = line.find('Hybrid SI') #will return location if string is found, otherwise returns -1
            idxCGS = line.find('Cgs') #will return location if string is found, otherwise returns -1
    
    if idxSI>idxCGS: #determine which unit string was found in the headerline and output
        return 'SI'
    else:
        return 'Cgs'
def parse_mass(file):
    """Function to extract sample from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    Mass in g or N/A
    """
    output = 'N/A'
    string = 'Mass' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=(line[idx+1:])  #value taken as everything to right of the setting name
        
            if output.find('N/A') > -1:
                output = 'N/A'
            else:
                output = float(output)

    return output
def measurement_times(file,Fk,Fj):
    """Function to estimate the time at which magnetization points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.
    Inputs:
    file: name of data file (string)    
    Fk: FORC indicies (int)
    Fj: Measurement indicies within given FORC
    Outputs:
    Ft: Estimated times at which the magnetization points were measured (float)
    """    
    unit=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if unit=='Cgs':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T

    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH

    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt3 = Hb2/alpha

    Npts=int(Fk.size)
    Ft=np.zeros(Npts)
    
    for i in range(Npts):
        if Fk[i]<=1+nc2:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*(Fk[i]-1)**2
        else:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*((Fk[i]-1)*(1+nc2)-nc2)

    return Ft
def parse_calibration(file):
    """Function to extract measured calibration points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    Hcal: sequence of calibration fields [float, SI units]
    Mcal: sequence of calibration magnetizations [float, SI units]
    tcal: Estimated times at which the calibration points were measured (float, seconds)
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## now need to pull out the calibration points, will be after alternate -9999.99 entries
    idxSAT = np.array(np.where(np.isin(H0, dum))) #location of dummy values
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSAT = idxSAT[0::2]+1 #every second index+1 should be calibration points

    Hcal=H0[idxSAT[0:-1]] #calibration fields
    Mcal=M0[idxSAT[0:-1]] #calibration magnetizations
    tcal=calibration_times(file,Hcal.size) #estimate the time of each calibratio measurement

    unit = parse_units(file)
    
    if unit=='Cgs': #ensure SI units
        Hcal=Hcal/1E4 #convert from Oe to T
        Mcal=Mcal/1E3 #convert from emu to Am^2

    return Hcal, Mcal, tcal
def calibration_times(file, Npts):
    """Function to estimate the time at which calibration points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.
    Inputs:
    file: name of data file (string)    
    Npts: number of calibration points (int)
    Outputs:
    tcal_k: Estimated times at which the calibration points were measured (float)
    """    
    unit=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if unit=='Cgs':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T
    
    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH
    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt2 = tr + tau + (Hc2-Hb2-dH)/alpha

    Npts=int(Npts)
    tcal_k=np.zeros(Npts)
    
    for k in range(1,Npts+1):
        if k<=1+nc2:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*(k-1)**2
        else:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*((k-1)*(1+nc2)-nc2)

    return tcal_k
def sample_details(fn):

    sample = fn.split('/')[-1]
    sample = sample.split('.')
    
    if type(sample) is list:
        sample=sample[0]

    units=parse_units(fn)
    mass=parse_mass(fn)
  
    return sample, units, mass
def measurement_limts(X):
    """Function to find measurement limits and conver units if required
    Inputs:
    file: name of data file (string)    
    Outputs:
    Hc1: minimum Hc
    Hc2: maximum Hc
    Hb1: minimum Hb
    Hb2: maximum Hb
    """    
    
    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(X["fn"],string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(X["fn"],string)

    string='Hc2' #upper Hc value for the FORC box
    Hc2=parse_header(X["fn"],string)

    string='Hc1' #lower Hc value for the FORC box
    Hc1=parse_header(X["fn"],string)

    if X['unit']=='Cgs': #convert CGS to SI
        Hc2=Hc2/1E4 #convert from Oe to T
        Hc1=Hc1/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T  

    return Hc1, Hc2, Hb1, Hb2

#### Unit conversion ####
def CGS2SI(X):
    
    X["H"] = X["H"]/1E4 #convert Oe into T
    X["M"] = X["M"]/1E3 #convert emu to Am2
      
    return X

#### low-level IO routines
def find_data_lines(fp):
    """Helper function to identify measurement lines in a FORC data file.
    
    Given the various FORC file formats, measurements lines are considered to be those which:
    Start with a '+' or,
    Start with a '-' or,
    Are blank (i.e. lines between FORCs and calibration points) or,
    Contain a ','
    Inputs:
    fp: file identifier
    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if ((line.startswith('+')) or (line.startswith('-')) or (line.strip()=='') or line.find(',')>-1.)]
def lines_that_start_with(string, fp):
    """Helper function to lines in a FORC data file that start with a given string
    
    Inputs:
    string: string to compare lines to 
    fp: file identifier
    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if line.startswith(string)]


#### PREPROCESSING OPTIONS ####
def options(X):
    style = {'description_width': 'initial'} #general style settings

    ### Define sample properties ###
    fn = X['fn']
    prop_title = widgets.HTML(value='<h3>Sample preprocessing options</h3>')
    mass_title = widgets.HTML(value='To disable mass normalization use a value of -1')

    sample, unit, mass = ut.sample_details(fn)

    sample_widge = widgets.Text(value=sample,description='Sample name:',style=style)
    
    if mass == "N/A":
        mass_widge = widgets.FloatText(value=-1, description = 'Sample mass (g):',style=style)
    else:
        mass_widge = widgets.FloatText(value=mass, description = 'Sample mass (g):',style=style)

    mass_widge1 = HBox([mass_widge,mass_title])
    
    ### Define measurement corrections ###
    correct_title = widgets.HTML(value='<h3>Select preprocessing options:</h3>')
    
    slope_widge = widgets.FloatSlider(
        value=70,
        min=1,
        max=100.0,
        step=1,
        description='Slope correction [%]:',
        style=style,
        readout_format='.0f',
    )
    
    slope_title = widgets.HTML(value='To disable high-field slope correction use a value of 100%')
    slope_widge1 = HBox([slope_widge,slope_title])
    
    drift_widge = widgets.Checkbox(value=False, description='Measurement drift correction')
    fpa_widge = widgets.Checkbox(value=False, description='Remove first point artifact')
    lpa_widge = widgets.Checkbox(value=False, description='Remove last point artifact')
    correct_widge = VBox([correct_title,sample_widge,mass_widge1,slope_widge1,drift_widge,fpa_widge,lpa_widge])

    preprocess_nest = widgets.Tab()
    preprocess_nest.children = [correct_widge]
    preprocess_nest.set_title(0, 'PREPROCESSING')
    display(preprocess_nest)

    X["sample"] = sample_widge
    X["mass"] = mass_widge
    X["unit"] = unit
    X["drift"] = drift_widge
    X["slope"] = slope_widge
    X["fpa"] = fpa_widge
    X["lpa"] = lpa_widge
    
    return X

#### PREPROCESSING COMMAND ####
def execute(X):
  
    #parse measurements
    H, Hr, M, Fk, Fj, Ft, dH = ut.parse_measurements(X["fn"])
    Hcal, Mcal, tcal = ut.parse_calibration(X["fn"])
    Hc1, Hc2, Hb1, Hb2 = ut.measurement_limts(X)
    
    # make a data dictionary for passing large numbers of arguments
    # should unpack in functions for consistency
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["dH"] = dH
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["Ft"] = Ft
    X["Hcal"] = Hcal
    X["Mcal"] = Mcal
    X["tcal"] = tcal
    X["Hc1"] = Hc1
    X["Hc2"] = Hc2
    X["Hb1"] = Hb1
    X["Hb2"] = Hb2

    if X['unit']=='Cgs':
        X = ut.CGS2SI(X)
    
    if X["drift"].value == True:
        X = drift_correction(X)   
  
    if X["slope"].value < 100:
        X = slope_correction(X)
  
    if X["fpa"].value == True:
        X = remove_fpa(X)
    
    if X["lpa"].value == True:
        X = remove_lpa(X)
    
    #extend FORCs
    X = FORC_extend(X)

    #perform lower branch subtraction
    X = lowerbranch_subtract(X)
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(121)
    X = plot_hysteresis(X,ax1)
    ax2 = fig.add_subplot(122)
    X = plot_delta_hysteresis(X,ax2)
    
    outputfile = X["sample"].value+'_HYS.eps'    
    plt.savefig(outputfile, bbox_inches="tight")
    plt.show()
    
    return X

#### PREPROCESSING ROUTINES ####
def remove_lpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Ft = X["Ft"]
    
    #remove last point artifact
    Nforc = int(np.max(Fk))
    W = np.ones(Fk.size)
    
    for i in range(Nforc):      
        Fj_max=np.sum((Fk==i))
        idx = ((Fk==i) & (Fj==Fj_max))
        W[idx]=0.0
    
    idx = (W > 0.5)
    H=H[idx]
    Hr=Hr[idx]
    M=M[idx]
    Fk=Fk[idx]
    Fj=Fj[idx]
    Ft=Ft[idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def remove_fpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    Ft = X["Ft"]
    
    #remove first point artifact
    idx=((Fj==1.0))
    H=H[~idx]
    Hr=Hr[~idx]
    M=M[~idx]
    Fk=Fk[~idx]
    Fj=Fj[~idx]
    Ft=Ft[~idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    Fj=Fj-1.
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def drift_correction(X):
  
    #unpack
    M = X["M"]
    Mcal = X["Mcal"]    
    Ft = X["Ft"]
    tcal = X["tcal"]
  
    #perform drift correction
    M=M*Mcal[0]/np.interp(Ft,tcal,Mcal,left=np.nan) #drift correction
  
    #repack
    X["M"] = M
  
    return X

def FORC_extend(X):
    
    Ne = 20 #extend up to 20 measurement points backwards
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    for i in range(int(X['Fk'][-1])):
        M0 = M[Fk==i+1]
        H0 = H[Fk==i+1]
        Hr0 = Hr[Fk==i+1][0]
        
        M1 = M0[0] - (np.flip(M0)[1:]-M0[0])
        H1 = H0[0] - (np.flip(H0)[1:]-H0[0])
            
        if M1.size>Ne:
            H1 = H1[-Ne-1:-1]
            M1 = M1[-Ne-1:-1]
        
        if i==0:    
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H1,H0))
            M_new = np.concatenate((M1,M0))
            Hr_new = np.ones(N_new)*Hr0
            Fk_new = np.ones(N_new)
            Fj_new = np.arange(N_new)+1-M1.size
        else:
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H_new,H1,H0))
            M_new = np.concatenate((M_new,M1,M0))
            Hr_new = np.concatenate((Hr_new,np.ones(N_new)*Hr0))
            Fk_new = np.concatenate((Fk_new,np.ones(N_new)+i))
            Fj_new = np.concatenate((Fj_new,np.arange(N_new)+1-M1.size))
            
    #pack up variables
    X['H'] = H_new
    X['Hr'] = Hr_new
    X['M'] = M_new
    X['Fk'] = Fk_new
    X['Fj'] = Fj_new
    
    return X

def lowerbranch_subtract(X):
    """Function to subtract lower hysteresis branch from FORC magnetizations
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    M: lower branch subtracted magnetization [float, SI units]
   
    
    """
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)


    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    X["H"] = H    
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr
    
    return X

    ###### HELPER FUNCTIONS TO READ FROM FILE

def slope_correction(X):
  
    #unpack
    H = X["H"]
    M = X["M"]
  
    # high field slope correction
    Hidx = H > (X["slope"]/100) * np.max(H)
    p = np.polyfit(H[Hidx],M[Hidx],1)
    M = M - H*p[0]
  
    #repack
    X["M"]=M
  
    return X



#### PLOTTING ROUTINES #####

def plot_hysteresis(X,ax):

  #unpack 
    M = X["M"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    #mpl.style.use('seaborn-whitegrid')
    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k')        

    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim,ylim])
  
    #ax.set_ylim([-1,1])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    xmax = np.max(np.abs(ax.get_xlim()))
    ax.set_xlim([-xmax,xmax])

    #label x-axis
    ax.set_xlabel('$\mu_0 H [T]$',horizontalalignment='right', position=(1,25), fontsize=12)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        ax.set_ylabel('$M [Am^2/kg]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
    else: 
        ax.set_ylabel('$M [Am^2]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

    
    X["xmax"]=xmax
    
    return X

def plot_delta_hysteresis(X,ax):

    #unpack 
    M = X["DM"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k') 
      
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
  
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim*0.1,ylim])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    Xticks = ax.get_xticks()
    Xidx = np.argwhere(np.abs(Xticks)>0.01)
    ax.set_xticks(Xticks[Xidx])

    xmax = X["xmax"]
    ax.set_xlim([-xmax,xmax])
    
    #label x-axis according to unit system
    ax.set_xlabel('$\mu_0 H [T]$',horizontalalignment='right', position=(1,25), fontsize=12)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        ax.set_ylabel('$M - M_{hys} [Am^2/kg]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
    else: 
        ax.set_ylabel('$M - M_{hys} [Am^2]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

    
    return X

def create_arrays(X, maxSF):

    Fk_int = (X['Fk'].astype(int)) #turn to int to use bincount
    counts = np.bincount(Fk_int) #time each no. appears in Fk (no. FORC on)
    max_FORC_len = np.max(counts) #max occurance of a FORC no. = longest FORC length = no. columns
    no_FORC = np.argmax(counts) #max FORC no.   = rows

    H_A = np.zeros((no_FORC, max_FORC_len)) #initialize arrays
    Hr_A = np.zeros((no_FORC, max_FORC_len))
    M_A = np.zeros((no_FORC, max_FORC_len))
    Fk_A = np.zeros((no_FORC, max_FORC_len))
    Rho = np.zeros((maxSF+1, no_FORC, max_FORC_len))
    #initialize zero values
    H_A[0,0] = X['H'][0]
    Hr_A[0,0] = X['Hr'][0]
    M_A[0,0] = X['M'][0]
    Fk_A[0,0] = X['Fk'][0]

    j=0 # just filled first point in first row
    i=0 # start at first row
    for cnt in range(1,len(X['Fk']+1)):
        if (X['Fk'][cnt] == X['Fk'][cnt-1]): #if Fk no is the same, stay on same row and fill data
            j +=1 #add one more to column and repeat
            H_A[i][j] = X['H'][cnt]
            Hr_A[i][j] = X['Hr'][cnt]
            M_A[i][j] = X['M'][cnt]     
        else:
            i +=1 #new row
            j = 0 #set column index back to zero
            H_A[i][j] = X['H'][cnt]
            Hr_A[i][j] = X['Hr'][cnt]
            M_A[i][j] = X['M'][cnt]            
        cnt +=1 #next point
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    X['rho'] = Rho
    X['no_FORC'] = no_FORC
    X['max_FORC_len'] = max_FORC_len
    return(X)

def calc_rho(X, SF):
    no_FORC = X['no_FORC']
    max_FORC_len = X['max_FORC_len']
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    M_A = X['M_A']
    Rho = X['rho']

    for i in range(no_FORC): #find main points, test without +1
        for j in range(max_FORC_len): #find each j indice
            #locate smoothing grids

            cnt = 0
            h1 = min(i, SF) #always on row go from SF below and SF above. no diffrence if this is i-1 and in k1 (j-1)
            h2 = min(SF, (no_FORC - i)) #try and loop over all points and not ignore boundaries,
            k1 = min(j, SF) #point to left, 1j if 0 etc or SF is in middle
            k2 = min(SF, (max_FORC_len-j)) #right hand side - either SF or if near edge do total - j (point at)

            A = np.zeros(((h2+h1+1)*(k1+k2+1),6))
            b = np.zeros(((h2+h1+1)*(k1+k2+1)))
            A[:,:] = np.nan
            b[:] = np.nan

            #if (M_A[i][j] != 0. and H_A[i][j] !=0 and Hr_A[i][j] != 0): 
            if (H_A[i][j] > Hr_A[i][j]):
                for h in range((-h1), (h2+1)): #loop over row in smoothing window
                    for k in range((-k1), (k2+1)): #loop over columns in smoothing window
                        if ((j+h+k) > 0 and (j+k+h) <= (max_FORC_len-1) and (i+h) > 0 and (i+h) <= (no_FORC -1)): 
                                #if (M_A[i+h][j+h+k] != 0. and H_A[i+h][j+h+k] !=0 and Hr_A[i][j+h+k] != 0): #remved but this makes a difference to plot
                            A[cnt, 0] = 1.
                            A[cnt, 1] = Hr_A[i+h][j+k+h] - Hr_A[i][j]
                            A[cnt, 2] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])**2.
                            A[cnt, 3] = H_A[i+h][j+k+h] - H_A[i][j]
                            A[cnt, 4] = (H_A[i+h][j+k+h] - H_A[i][j])**2.
                            A[cnt, 5] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])*(H_A[i+h][j+k+h] - H_A[i][j])
                            b[cnt] = M_A[i+h][j+k+h]

                            cnt+=1 #count number values looped over
                    #print('A', A)
                    #print('b', b)
                A = A[~np.isnan(A).any(axis=1)]
                b = b[~np.isnan(b)]
                    #print('A no nan', A)
                    #print('b no nan', b)
                if (len(A)>2): #min no. points to need to smooth over
                        #cmatrix = np.matmul(np.transpose(A), A)
                    dmatrix, res, rank, s = lstsq(A,b)
                    Rho[SF][i][j] = (-1.*(dmatrix[5]))/2.

                else:
                    Rho[SF][i][j] = 0.
            else:
                Rho[SF][i][j] = 0.
            j +=1
        i += 1

    X['H_A'] = H_A #repack variables
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    X['rho'] = Rho
    X['no_FORC'] = no_FORC
    X['max_FORC_len'] = max_FORC_len
    return(X)
    
    
def nan_values(X, maxSF):
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    Rho = X['rho']
    for i in range(len(H_A)):
        for j in range(len(Hr_A[0])):
            if (H_A[i][j] == 0.0):
                H_A[i][j] = 'NaN'
                Hr_A[i][j] = 'NaN'
                
    for k in range(maxSF+1):
        for i in range(len(H_A)):
            for j in range(len(Hr_A[0])):
                if (Rho[k][i][j] == 0.0):
                    Rho[k][i][j] = 'NaN'
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['rho'] = Rho
    return(X)
    
def plot_pre_rotate_FORC(X, SF):
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    Rho = X['rho']
    fig1, ax2 = plt.subplots(constrained_layout=True)
    CS = ax2.contour(H_A, Hr_A, Rho[SF], 50, cmap='rainbow')
    plt.xlim(-0.1, 0.25)
    plt.ylim(-0.5, 0.15)
    plt.colorbar
    plt.xlabel('H_applied')
    plt.ylabel('H_reversal')
    plt.title('Pre-rotated FORC diagram, SF = %d' %SF)
    plt.show
    
    
def rotate_FORC(X):
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    Hc= (H_A - Hr_A)/2. #x axis
    Hu = (H_A + Hr_A)/2. #y axis
    X['Hc'] = Hc
    X['Hu'] = Hu
    return(X)  
    
def plot_simple_FORC(X, SF, sample_name):
    Hc = X['Hc']
    Hu = X['Hu']
    Rho = X['rho']
    plt.contourf(Hc, Hu, Rho[SF], 20, cmap='rainbow', alpha = 0.7)
    plt.contour(Hc, Hu, Rho[SF], 20, cmap='rainbow')
    plt.xlabel('Hc (Bc) (T)')
    plt.ylabel('Hu (Bu) (T)')
    plt.xlim(0, 0.2)
    plt.ylim(-0.1, 0.06)
    plt.colorbar()
    plt.title('rotated FORC diagram, SF = %d, sample %s ' %(SF , sample_name))
    plt.show
    plt.savefig('FORC_diagram_mh_4.png') #how to add file name issues - take details from irop - add sample name etc into it


def norm_rho(X, SF):
    Rho = X['rho']
    Rho_n = np.copy(Rho)
    max_Rho = np.nanmax(Rho_n)
    i = 0
    j = 0
    for i in range(len(Rho[0])):
        for j in range(len(Rho[1])):
            Rho_n[SF][i][j] = Rho[SF][i][j]/max_Rho
    X['rho_n'] = Rho_n
    X['max_Rho'] = max_Rho
    return(X)


def plot_general_FORC_basic(x, y, z, SF, sample_name):
    z = z[SF]
    #need to edit labels from input
    plt.contourf(x, y, z, 20, cmap='rainbow', alpha = 0.7)
    plt.contour(x, y, z, 20, cmap='rainbow')
    plt.xlabel('Hc (Bc) (T)')
    plt.ylabel('Hu (Bu) (T)')
    plt.xlim(0, 0.2)
    plt.ylim(-0.1, 0.07)
    cbar = plt.colorbar()
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar.set_label('normalized Rho', rotation=270)
    plt.title('rotated FORC diagram, SF = %d, sample %s ' %(SF , sample_name))
    plt.show
    plt.savefig('FORC_diagram_mh_SF_4_norm.png')
    
def FORCinel_colormap(Z):

    #setup initial colormap assuming that negative range does not require extension
    cdict = {'red':     ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                         (0.3193,  102/255, 102/255),
                       (0.563,  204/255, 204/255),
                       (0.6975,  204/255, 204/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255)),

            'green':   ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  178/255, 178/255),
                        (0.563,  204/255, 204/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  102/255, 102/255),
                       (0.9748,  25/255, 25/255),
                       (1.0, 25/255, 25/255)),

             'blue':   ((0.0,  255/255, 255/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  102/255, 102/255),
                        (0.563,  76/255, 76/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255))}

    if np.abs(np.min(Z))<=np.nanmax(Z):#*0.19: #negative extension is not required
        #cmap = LinearSegmentedColormap('forc_cmap', cdict)
        vmin = -np.nanmax(Z)#*0.19
        vmax = np.nanmax(Z)
    else: #negative extension is required
        vmin=np.nanmin(Z)
        vmax=np.nanmax(Z)        
    
    anchors = np.zeros(10)
    anchors[1]=(-0.025*vmax-vmin)/(vmax-vmin)
    anchors[2]=(-0.005*vmax-vmin)/(vmax-vmin)
    anchors[3]=(0.025*vmax-vmin)/(vmax-vmin)
    anchors[4]=(0.19*vmax-vmin)/(vmax-vmin)
    anchors[5]=(0.48*vmax-vmin)/(vmax-vmin)
    anchors[6]=(0.64*vmax-vmin)/(vmax-vmin)
    anchors[7]=(0.80*vmax-vmin)/(vmax-vmin)
    anchors[8]=(0.97*vmax-vmin)/(vmax-vmin)
    anchors[9]=1.0

    Rlst = list(cdict['red'])
    Glst = list(cdict['green'])
    Blst = list(cdict['blue'])

    for i in range(9):
        Rlst[i] = tuple((anchors[i],Rlst[i][1],Rlst[i][2]))
        Glst[i] = tuple((anchors[i],Glst[i][1],Glst[i][2]))
        Blst[i] = tuple((anchors[i],Blst[i][1],Blst[i][2]))
        
    cdict['red'] = tuple(Rlst)
    cdict['green'] = tuple(Glst)
    cdict['blue'] = tuple(Blst)

    cmap = colors.LinearSegmentedColormap('forc_cmap', cdict)

    return cmap, vmin, vmax

#@jit
def plot_FORCinel(X, SF, sample_name):
    Hc = X['Hc']
    Hu = X['Hu']
    Rho_n = X['rho_n']
    Hc2 = X['Hc2']
    Hb1 = X['Hb1']
    Hc2 = X['Hc2']
    Hb2 = X['Hb2']
    xmin = 0
    xmax = np.round(Hc2*1000)/1000
    ymin = np.round((Hb1-Hc2)*1000)/1000
    ymax = np.round(Hb2*1000)/1000
    contour = 50
    xlabel_text = '$\mu_0 H_c [T]$' #label Hc axis [SI units]
    ylabel_text = '$\mu_0 H_u [T]$' #label Hu axis [SI units]
    cbar_text = '$Am^2 T^{-2}$'
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
     #plt.contour(Hc, Hu, Rho_n, 20, cmap='rainbow')   
    cmap, vmin, vmax = FORCinel_colormap(Rho_n) #runs FORCinel colormap
    CS = ax.contourf(Hc, Hu, Rho_n[SF], cmap = cmap, vmin=vmin, vmax=vmax) #plots using FORCinel colours
    cbar = fig.colorbar(CS,fraction=0.04, pad=0.08)
    cbar.ax.tick_params(labelsize=14)
            #cbar.ax.set_title(cbar_text,fontsize=14)
    cbar.set_label(cbar_text,fontsize=14)

    CS2 = ax.contour(CS, levels = CS.levels[::2], colors='k',linewidths=0.8)

    ax.set_xlabel(xlabel_text,fontsize=14) #label Hc axis [SI units]
    ax.set_ylabel(ylabel_text,fontsize=14) #label Hu axis [SI units]  

        # Set plot Xlimits
    xlimits = np.sort((xmin,xmax))
    ax.set_xlim(xlimits)

        #Set plot Ylimits
    ylimits = np.sort((ymin,ymax))
    ax.set_ylim(ylimits)

        #Set ticks and plot aspect ratio
    ax.tick_params(labelsize=14)
    ax.set_aspect('equal') #set 1:1 aspect ratio
    ax.minorticks_on() #add minor ticks
    #"'{0}' is longer than '{1}'".format(name1, name2)
    plt.title("rotated FORC diagram, sample '{0}'".format(sample_name))
    plt.show
    
    
def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
#@jit
def half_max_x(x, y, ym):
    half = ym/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

#@jit
def find_fwhm(X, SF, sample_name, fwhmlist): #do for 1 SF - 
    Rho = X['rho'] #in functins
    Hu = X['Hu']
    #poss add in loop 
    indices = np.unravel_index(np.nanargmax(Rho[SF]),Rho[SF].shape)
    fwHu = []
    fwRho = []
    for i in range(len(Rho[SF])):
        fwHu.append(Hu[i][indices[1]]) #add in SF
        fwRho.append(Rho[SF][i][indices[1]])
        i+=1

    fwHu = np.array(fwHu)
    fwRho = np.array(fwRho)
    fwHu = fwHu[~np.isnan(fwHu)]
    fwRho = fwRho[~np.isnan(fwRho)] #have my arrays for fwhm calc
    r0 = -1
    r1 = 1

    
    #here adjust size 
    loc_o = np.argmin(abs(fwHu))
    fwHu_f = fwHu[:loc_o] #loc zero needed
    fwRho_f = fwRho[:loc_o] #loc zero needed

    loc_m = np.argmin(abs(fwRho_f))
    fwHu_c = fwHu[loc_m:(loc_o +(loc_o - loc_m))] #loc zero needed
    fwRho_c = fwRho[loc_m:(loc_o +(loc_o - loc_m))] #loc zero needed

    plt.plot(fwHu_c, fwRho_c, label = SF)
    plt.show
    m_rho_a = np.sort(fwRho_c)
    i = 1
    while ((r0 <0) or (r1 > 0)): #opposte to FWHM crossing 0 
        ym = m_rho_a[-i]
        # find the two crossing points
        try:
            hmx = half_max_x(fwHu_c,fwRho_c, ym)
            
            r0 = hmx[0]
            r1 = hmx[1]
            fwhm = hmx[0] - hmx[1]
        except:
            print('Error in calculating FWHM for SF %',SF)
            pass
        
        if (i >3):
            print('too noisy')
            fwhm = 'Nan'
            break
        i+=1
  
    fwhmlist.append(fwhm)

    half = max(fwRho_c)/2.0
    plt.plot(hmx, [half, half], label = SF)
    plt.xlabel('Hu')
    plt.ylabel('Rho')
    plt.legend()
    
    plt.title("FWHM plot for sample '{0}'".format(sample_name))
    plt.show
    

def plot_fwhm(SFlist, fwhmlist, X):
    st_line_SFlist = []
    polyfwhm = []
    polySF = []
    print(X['maxSF1'])
    maxSF1 = X['maxSF1']
    for i in range(maxSF1+1):
        st_line_SFlist.append(i)
        i +=1

    st_line_SFlist= np.array(st_line_SFlist)
    SFlist = np.array(SFlist)
    fwhmlist = np.array(fwhmlist)

    for i in range(len(fwhmlist)):
        if (fwhmlist[i] != 'Nan'):
            polyfwhm.append(float(fwhmlist[i]))
            polySF.append(float(SFlist[i]))

    plt.scatter(polySF, polyfwhm)
    plt.xlim(0,5.3)
    plt.ylim(0, 0.045)

    b, m = polyfit(polySF, polyfwhm, 1)
 
    plt.xlabel('SF')
    plt.ylabel('FWHM')
    plt.plot(st_line_SFlist, b + m * st_line_SFlist, '-')
    plt.show 
      
    Hu = X['Hu']

    i=0
    for i in range(len(fwhmlist)):
        if (fwhmlist[i] == 'Nan'):

            fwhmlist[i] = float(m*SFlist[i] + b)

    fwhmlist = np.array(fwhmlist)
    fwhmlist = fwhmlist.astype(np.float) 

    #print(maxSF)
    while True:
        sf_choose = (input("Pick a SF between 2 and 5 to calculaute Sf=0 from:" ))
        print(type(sf_choose))
        try:
            sf_choose = int(sf_choose)
            if (sf_choose >= 2) and (sf_choose <= maxSF1):
                print('in bounds')
                break
        except ValueError:
            print('Not an interger')
            True
       # if (isinstance(sf_choose, int)):
        print('int')
  
 
    print ("Is this what you just said?", sf_choose)
    print(sf_choose)

    X['sf_choose'] = sf_choose
    Hu_0 = Hu*(b/fwhmlist[sf_choose-2])
    sf_correct = (b/fwhmlist[sf_choose-2])
    X['sf_correct'] = sf_correct #switch back to 2
    X['Hu_0'] = Hu_0
    
    
def check_fwhm(SFlist, fwhmlist, X):
    answer = None
    maxSF1 = X['maxSF1']
    while answer not in ("yes", "no"):
        answer = input("Are any the FWHM unreliable? Enter yes or no: ")
        if answer == "yes":
             #rest of code
            while True:
                sf_pick = (input("Which SF is unrealiable and needs to be removed?:" ))
                print(type(sf_pick))
                try:
                    sf_pick = int(sf_pick)
                    if (sf_pick >= 2) and (sf_pick <= maxSF1):
                        print('in bounds')
                        break
                except ValueError:
                    print('Not an interger')
                    True
       # if (isinstance(sf_choose, int)):
            print('int')
            #code
            val = int(sf_pick)
            #SFlist.remove(val)
            fwhmlist[val-2] = 'Nan'
            print('fwhmlist', fwhmlist)
            print('Sflist', SFlist)
            
                
        elif answer == "no":
            print('points all ok')
            break #stop functon ? print 'points ok'
        else:
            print("Please enter yes or no.")
    plot_fwhm(SFlist, fwhmlist, X) #give Hu_0
    return
        

def divide_mu0(X):
    mu0 = mu0=4*pi*1e-7
    X['Hc_mu'] = X['Hc']/mu0
    X['Hu_0_mu'] = X['Hu_0']/mu0
    X['Hu_mu'] = X['Hu']/mu0
    return(X)

def FORCinel_colormap(Z):

    #setup initial colormap assuming that negative range does not require extension
    cdict = {'red':     ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                         (0.3193,  102/255, 102/255),
                       (0.563,  204/255, 204/255),
                       (0.6975,  204/255, 204/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255)),

            'green':   ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  178/255, 178/255),
                        (0.563,  204/255, 204/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  102/255, 102/255),
                       (0.9748,  25/255, 25/255),
                       (1.0, 25/255, 25/255)),

             'blue':   ((0.0,  255/255, 255/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  102/255, 102/255),
                        (0.563,  76/255, 76/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255))}

    if np.abs(np.min(Z))<=np.nanmax(Z):#*0.19: #negative extension is not required
        #cmap = LinearSegmentedColormap('forc_cmap', cdict)
        vmin = -np.nanmax(Z)#*0.19
        vmax = np.nanmax(Z)
    else: #negative extension is required
        vmin=np.nanmin(Z)
        vmax=np.nanmax(Z)        
    
    anchors = np.zeros(10)
    anchors[1]=(-0.025*vmax-vmin)/(vmax-vmin)
    anchors[2]=(-0.005*vmax-vmin)/(vmax-vmin)
    anchors[3]=(0.025*vmax-vmin)/(vmax-vmin)
    anchors[4]=(0.19*vmax-vmin)/(vmax-vmin)
    anchors[5]=(0.48*vmax-vmin)/(vmax-vmin)
    anchors[6]=(0.64*vmax-vmin)/(vmax-vmin)
    anchors[7]=(0.80*vmax-vmin)/(vmax-vmin)
    anchors[8]=(0.97*vmax-vmin)/(vmax-vmin)
    anchors[9]=1.0

    Rlst = list(cdict['red'])
    Glst = list(cdict['green'])
    Blst = list(cdict['blue'])

    for i in range(9):
        Rlst[i] = tuple((anchors[i],Rlst[i][1],Rlst[i][2]))
        Glst[i] = tuple((anchors[i],Glst[i][1],Glst[i][2]))
        Blst[i] = tuple((anchors[i],Blst[i][1],Blst[i][2]))
        
    cdict['red'] = tuple(Rlst)
    cdict['green'] = tuple(Glst)
    cdict['blue'] = tuple(Blst)

    cmap = matplotlib.colors.LinearSegmentedColormap('forc_cmap', cdict)
    
    return cmap, vmin, vmax
    

def plot_general_FORC_basic1(x, y, z, SF, sample_name):
    z = z[SF]
    print(sample_name)
    con = np.linspace(0.1, 1, 9)

    #need to edit labels from input
    cmap, vmin, vmax = FORCinel_colormap(z) #runs FORCinel colormap
    
    #FORCinel_colormap(z)
    plt.contourf(x, y, z, 50, cmap= cmap, fontsize=14)
    cbar = plt.colorbar()
    plt.contour(x, y, z, con, colors = 'k', fontsize=14)
    plt.xlabel('Hc (T)', fontsize=14)
    plt.ylabel('Hu (T)', fontsize=14)
    plt.xlim(0, 0.1)
    plt.ylim(-0.03, 0.04)
    plt.tick_params(axis='both', which='major', labelsize=14)
    #cbar = plt.colorbar(heatmap)
    #cbar.ax.set_yticklabels(['0','1','2','>3'])
    cbar.set_label('FORC distribution (normalised)', rotation=270, fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=14)
    plt.title("Normalised FORC diagram for sample '{0}', using smoothing factor '{1}'".format(sample_name, SF))
    plt.tight_layout()
    plt.show
    plt.savefig('FORC_dia_sample{0}_sf_{1}.png'.format(sample_name, SF))
    return 
 
def inter_FORC(X):
    Hu_f = X['Hu_mu'].flatten() #change Hu_0 to Hu
    Hc_f = X['Hc_mu'].flatten()
    Rho_f_2 = X['rho'][X['sf_choose']].flatten() #flatten SF 3 section
    
    #remove nan
    Hu_f = Hu_f[~np.isnan(Hu_f)]
    Hc_f = Hc_f[~np.isnan(Hc_f)]
    Rho_f_2 = Rho_f_2[~np.isnan(Rho_f_2)]
    
    step_xi = np.nanmax(X['Hc_mu'])/181.
    step_yi = (np.nanmax(X['Hu_mu']) - np.nanmin(X['Hu_mu']))/146.

    # target grid to interpolate to
    xi = np.arange(0,np.nanmax(X['Hc_mu']),step_xi) #same size needed but not same axes - thinkt and chck
    yi = np.arange(np.nanmin(X['Hu_mu']),np.nanmax(X['Hu_mu']),step_yi) #same size needed but not same axes - thinkt and chck
    xi1,yi1 = np.meshgrid(xi,yi) 


    # interpolate
    zi = griddata((Hc_f,Hu_f),Rho_f_2,(xi1,yi1),method='cubic') 

    X['xi1'] = xi1
    X['yi1'] = yi1
    X['zi'] = zi
    return (X)
    
def inter_rho(xi_s_f, yi_s_f, zi_s_f, hys, i): # when call use xi_s etc, i is no. hysteron to test
    xi1_row = xi_s_f[0,:] #should be different
    #print(xi1_row)
    #print(hys[i,0])
    up_hc = xi1_row[xi1_row > hys[i,0]].min()   #test on first data point do with 1D array, value is slightly above
    lo_hc = xi1_row[xi1_row < hys[i,0]].max() #works
    #print(up_hc,lo_hc)
    up_hc_idx = list(xi1_row).index(up_hc) #correct
    lo_hc_idx = list(xi1_row).index(lo_hc)
    #print(up_hc_idx, lo_hc_idx)
    yi1_col = yi_s_f[:,0] #should be different

    up_hi = yi1_col[yi1_col > hys[i,1]].min()   #test on first data point do with 1D array, value is slightly above
    lo_hi = yi1_col[yi1_col < hys[i,1]].max() #works - doesnt work on all dta points

    up_hi_idx = list(yi1_col).index(up_hi) #correct
    lo_hi_idx = list(yi1_col).index(lo_hi)

    x_arr = np.array([xi_s_f[lo_hi_idx,lo_hc_idx], xi_s_f[up_hi_idx, lo_hc_idx], xi_s_f[up_hi_idx, up_hc_idx], xi_s_f[lo_hi_idx, up_hc_idx]])
    y_arr = np.array([yi_s_f[lo_hi_idx,lo_hc_idx], yi_s_f[up_hi_idx, lo_hc_idx], yi_s_f[up_hi_idx, up_hc_idx], yi_s_f[lo_hi_idx, up_hc_idx]])
    z_arr = np.array([zi_s_f[lo_hi_idx,lo_hc_idx], zi_s_f[up_hi_idx, lo_hc_idx], zi_s_f[up_hi_idx, up_hc_idx], zi_s_f[lo_hi_idx, up_hc_idx]])

    f = interp2d(x_arr, y_arr, z_arr, kind='linear')
    
    hys[i,3] = f(hys[i,0], hys[i,1]) #swtiched round
    #print('hys[i,3]', hys[i,3])
    return hys
    
    
def sym_FORC(X):
    xi1 = X['xi1']
    yi1 = X['yi1']
    zi = X['zi']
    yi_axis = np.copy(yi1)

    yi_axis = abs(yi_axis) - 0

    
    indices = np.unravel_index(np.nanargmin(yi_axis),yi_axis.shape)
   

    xi_s = np.copy(xi1)
    yi_s = np.copy(yi1)
    zi_s = np.copy(zi)
  
    x=1
    j=0

    while x < (len(xi1) - indices[0]): #do for rows between max and dist go upto - 177

        j=0
        for j in range(len(xi_s[0])):
            #print(x,j)
            #print(indices[0]-x, indices[0]+x) #(471 - and 471 - x)

            find_mean = np.array([zi_s[indices[0]+x][j],zi_s[indices[0]-x][j]])
            #print('find_mean',find_mean)
            find_mean = find_mean[~np.isnan(find_mean)]
            zi_s[indices[0]+x][j] = np.mean(find_mean)
            #print('zi_s1', zi_s[indices[0]+x][j])
            zi_s[indices[0]-x][j] = zi_s[indices[0]+x][j]
            #print('zi_s2', zi_s[indices[0]-x][j])
            j+=1
        x+=1        
    
    lower_x_bound = indices[0] - (len(xi_s) - indices[0])
    upper_x_bound = len(xi_s)
    

    xi_s_cut = xi_s[lower_x_bound:upper_x_bound,:]
    yi_s_cut = yi_s[lower_x_bound:upper_x_bound,:]
    zi_s_cut = zi_s[lower_x_bound:upper_x_bound,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi_s_cut,yi_s_cut,zi_s_cut) #lost nparange arguement - see what does

    plt.xlabel('Hc',fontsize=16)
    plt.ylabel('Hu',fontsize=16)
    plt.title("symmetrical FORC diagram (sample '{0}') with SF '{1}'".format(X['name'], X['sf_choose']))
    plt.colorbar()
    plt.show
    
    X['xis'] = xi_s_cut
    X['yis'] = yi_s_cut
    X['zis'] = zi_s_cut
    
    
    return(X)
    
    
def norm_z(X):

    z_pre_norm = X['zis']

    maxval_z = np.nanmax(z_pre_norm)

    minval_z = np.nanmin(z_pre_norm)

    z_norm = (z_pre_norm)/(maxval_z)

    X['zis'] = z_norm

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(X['xis'],X['yis'],X['zis']) #lost nparange arguement - see what does

    plt.xlabel('Hc',fontsize=16)
    plt.ylabel('Hu',fontsize=16)
    plt.title("Normalised symmetrical FORC diagram for sample '{0}' with SF '{1}'".format(X['name'], X['sf_choose']))
    plt.colorbar()
    plt.show
    return(X)
    
def hys_angles():
    
    angle = random.random()
    #angle = 0.5
    phi = acos(2*angle - 1) #bnot just between 0 and 1.6 becasue times it by 2 and -1. therefore always less than 2, gives negative range too
    if(phi > (pi/2.)): 
        phi = pi-phi
    angle2 = random.random()
    phistatic = acos(2*angle2 - 1)
    if(phistatic > (pi/2.)):
        phistatic = pi-phistatic
    
    angle3 = random.random()
    thetastatic = 2*pi*angle3
    return phi, phistatic, thetastatic
    
def calc_hk_arrays(hys, num, V): #this is called for each inv point, can do at once? num is num_pop
    tm = V['tm']
    ms = V['ms']
    hfsum = 0
    hc = hys[:,0] #this could be array hc = hys[:,0]
    tempt = 300.
    
    hf = (hc**(eglip))*(10**(-0.52)) #uses hc not Hc - not changed yet in adrians code i dont think, hf could be array
 
    phi = hys[:,5] #this could be an array phi = hys[:,5]
    
    phitemp = (((np.sin(phi))**(2./3.))+((np.cos(phi))**(2./3.)))**(-3./2.) #phi = hys[i,5], phitemp could be array

    gphitemp = (0.86 + (1.14*phitemp)) #gphitemp could be array
  
    hatmp = hc/(sqrt(2)) #this is the hc once its been divided could be array
  
    print('tm for ht calc', tm)
    ht = hf*(log(tm/tau)) # tau is 10E-9, ht>0 as tm>tau. could be array, tm, tau constants
    
    hktmp = hatmp +ht + (2*hatmp*ht+ht**2)**0.5 #first guess for iteration, its a max guess, can be array
    hktmpmax = hktmp
    
    
    hktmpstore = hktmp
    i=0
    for i in range(int(num)):
        factor = 0.5
        searchme = 1000000.0     
        hstep = (hktmp[i]-hatmp[i])/5.

        while (abs(searchme)> 5):
            searchme = hktmp[i] - hatmp[i] - hktmp[i]*((2*ht[i]*phitemp[i]/hktmp[i])**(1/gphitemp[i])) #
            hktmpstore[i] = hktmp[i]
            #print(abs(searchme))
            if (searchme > 0):
                hktmp[i] = hktmp[i] - hstep
            else:
                hktmp[i] = hktmp[i] + hstep
                hstep = hstep*factor #this else should be for the Hktmp = hktmp - hstep

    hkphi = hktmpstore #think function before made arrys so just do array opertatiosn now
    hk = hkphi/phitemp

    hys[:,9] = hk #unsure if this will it fill correctly
    kb=1.3806503e-23 
    vol = (kb*tempt)/(hf*mu0*ms)
    hys[:,10] = vol
    
    return(hys)
    
    
def pop_hys(num_hys, X, V): #this function will populate all from beggining each time
    #populate hysterons
    #num_hys = 10
    hys = np.zeros((num_hys,11)) 
    
    #Hc, Hi, random no (Rho test), rho_frominter_test, mag (1), angle, static angle, mag times angle, Hf, vol
    num_pop = num_hys/2
    
    #maxHc,minHc,maxHi,MinHc
    #within symmetrical part - min is zero for both
    minHc = 0
    minHi = 0
    xi_s_cut = X['xis']
    yi_s_cut = X['yis']
    zi_s_cut = X['zis'] 
    maxHc = np.nanmax(xi_s_cut) #change to hu values, change to cut limits and sometmes slightly out and stopped 
    maxHi = np.nanmax(yi_s_cut)
    maxHi = X['Hb2'] #overwrite 
    maxHc = X['Hc2']
    #print(minHc, maxHc, minHi, maxHi)
    #loop over each hysteron to fill it with a value
    #want to do 5 values
    i=0
    while (i < int(num_pop)):
        z1 = random.random()
        z2 = random.random()
        z3 = random.random()
  
        hys[i,0] = (z2*maxHc)/mu0

        hys[i,1] = (z3*maxHi)/mu0
        

        hys = inter_rho(xi_s_cut, yi_s_cut, zi_s_cut, hys, i) #ztestnorm
        hys[i,1] = hys[i,1]*X['sf_correct']
        hys[i,5], hys[i,6], hys[i,7] = hys_angles() #calc for half hys
      
        if ((hys[i,1]) <= (hys[i,0])) and (hys[i,3] >= 0) and (hys[i,3] <= 1): #add in if hys[i,3] is positive - may not need
            i +=1 #cna have if statements below which can sto pthis #this line to change - remove hys[i,2] <= hys[i,3]
        
    hys = calc_hk_arrays(hys, int(num_pop), V) # try to calc hk using arrrys 
    hys[:,4] = 1
 
    hys[:,8] = hys[:,5]*hys[:,4] #move to be more efficent
    num_pop = int(num_pop)
    j=0
    for j in range(num_pop):
        hys[(j+num_pop),:] = hys[j,:]
        hys[j+num_pop,1] = -hys[j,1]
        j+=1

    for k in range(num_hys): #add in edit of phi static
        xstatic=sin(hys[k,6])*cos(hys[k,7])
        ystatic=sin(hys[k,6])*sin(hys[k,7])
        zstatic=cos(hys[k,6])
        xstatic=xstatic/sqrt(xstatic**2+ystatic**2+zstatic**2)
        ystatic=ystatic/sqrt(xstatic**2+ystatic**2+zstatic**2)
        zstatic=zstatic/sqrt(xstatic**2+ystatic**2+zstatic**2)

        hys[k,6]=acos(max(abs(xstatic),abs(ystatic),abs(zstatic)))  
    return hys, num_pop
    
    
def af_nrm_file():
    af_n = FileChooser()
    display(af_n)
    
    print('done')
    return(af_n)
    
    
def af_irm_file():
    af_i = FileChooser()
    display(af_i)

    return(af_i)
    
def demag_data(X):
    af_step = []
    af_nrm_n = []
    af_sirm_n = []
    af_step_irm = []

    af_nrm_data = open(X['af_n_fn'], "r") #write file
    for myline1 in af_nrm_data:
        line = myline1.split(' ') #split into elements by spaces

        af_step.append(float(line[1])) 
       # print('af_step', af_step)
        af_nrm_n.append(float(line[3])) 
        #print('af_nrm_n', af_nrm_n)

    af_nrm_data.close()

    af_irm_data = open(X['af_i_fn'], "r") #write file
    for myline in af_irm_data:
        line = myline.split(' ') #split into elements by spaces
    
        af_sirm_n.append(float(line[3])) 
        #print(af_nrm)

    af_irm_data.close()

    #no af steps
    cntfield = len(af_step)

    af_step = np.array(af_step)
    af_nrm_n = np.array(af_nrm_n)
    af_sirm_n = np.array(af_sirm_n)

    af_sirm_n = af_sirm_n[:len(af_nrm_n)]

    afnorm = af_nrm_n[0]/af_sirm_n[0] #only used for write out if want to use need to make in X
    #af_step = af_step/10.
    X['af_step'] = af_step
    X['af_nrm'] = af_nrm_n
    X['af_irm'] = af_sirm_n
    X['cntfield'] = cntfield
    return(X)

def blockfind(temp, field, afswitch, V, X): #try removing counttime as input/output
    hys = V['hys']
    num_hyss = V['num_hyss']
    hcstore = V['hcstore']
    histore = V['histore']
    beta = V['beta']
    rate = V['rate']
    aconst = V['aconst']
    tempmin = V['tempmin']
    sense = V['sense']
    
    tm = V['tm'] #try setting this for use in AF demag
    hcless = 0
    hcmore = 0
    memory = 0
    hfstore = np.zeros(num_hyss)
    #totalblocktwo = 0 #line 703
    #totalblock = 0 #line 703
    ms = ms0*beta #does beta need to be global or if its just used its ok
    conv=(mu0*1000)/(sqrt(2))
    #print(counttime)
    af_step = X['af_step']
    counttime = V['counttime']
    blockper = V['blockper'] 
    countiold = V['countiold']
    totalblocktwo = V['totalblocktwo']
    totalblock = V['totalblock']
    blocktemp = V['blocktemp'] 
    boltz = V['boltz']
    blockg = V['blockg']
    totalmoment = V['totalmoment']
    #print(V['blockg'])
    
    aftotalm = V['aftotalm']
    sir = V['sir']
    print('sir', sir)
    totalm = V['totalm']
    #sir = V['sir'] # added in
    counttime += 1 #whats counttime originally, added one on on line 916. start as zero. line 99. needs to be global as changing value
    samplemax = num_hyss
    for spfind in range(1): #do once?
        volsp = 0.0
        forccount =0.0
        counti = 0
        for i in range(num_hyss): #when run use num_hyss, check this should be indented

            if (blocktemp[i] == 0): #first go they are all 1, means blocked?
                
                volsp = volsp + (hys[i,10]*hys[i,3]) # for the forc, forcdist (hys(3)-random no. set, hys(4)- one at diag pos - seems one in adrians but i did diff ave)
               
                counti +=1 #add each onto number i looked at, num hys gone through
                forccount = forccount + 1 #forccount - add up sum rho, done one n forc diag
                
        if (forccount > 0): #should always be if blocktemp = 0
            meanvolsp = volsp/forccount #ones not blocked are sp, mean is vol/forccount - mag?, sum works

        else:
            meanvolsp = 0
        
        spcounti = counti #no hys sp is number count blocktemp 0 for = 10

        counti = 0 #makecount 0 again -start next bit
        tmmean = 0.0

        for i in range(num_hyss): #dynamic contributions - to hc dormann 1988 used?
  
            phi1 = pi*30/180. #give random angle 
            #print('i', i, 'spcounti', spcounti, 'counti', counti)
            n = 1 #no nearest neighbours
           
            deltahc = 0 #set all to zero? or just 1 valuey, is this correct? why reset to zero, not made any difference
           
            hc=(sqrt(2))*(hys[i,9]*beta+deltahc*spcounti/samplemax) #test and time HCbysqrt2 to get hc and matfhes similar order mag to hys[i,0/mu0 -> hc in same units using in hk - seems correct]
           
            hcstore[i] = hc/(sqrt(2)) #different to hys[0]
        
            
            hi = (sqrt(2))*hys[i,1]*beta*blockper #divide here by mu0
            
    
            histore[i] = hi/(sqrt(2)) #should all be zero, without blockper, all look milar to hys[1,0] in mg
          
         
            phitemp=((sin(hys[i,5])**(2./3.))+(cos(hys[i,5])**(2./3.)))**(-3./2.) #phitemp from hys[5] -> phi
       
            #print('phitemp', phitemp)
            g=0.86+1.14*phitemp
 
            hf=((hys[i,9]**eglip))*(10**(-0.52+eglid))*temp/(300*beta) #where this eq come from
       
            hfstore[i] = hf #values at expected as hc/mu0
         
      
            
            if (rate == 1): #rate first set to 1 so shold do this
               
                tm=(field*temp*aconst)/((temp-tempmin)*hf) # nb both hf and field *2**.5
      
                tmmean=tmmean+tm # large number add up all tm 
                tmstore=tm
                
                if (tm == 0.0): #unsure
                    tm = 60.0
 
            ht = (roottwohffield)*hf*(log(tm/tau)) #new tm 
       
            if (temp < 0):
                #write to screen
                print('ht', ht*mu0*1000, 'hf', hf*mu0*1000, '0.54' ,hc*mu0*1000, tm) #allow check with adrians line 1567
            #for each i calc bracket, and hi flip etc
            bracket = 1-(2*ht*phitemp/hc)**(1/g)
            
            hiflipplus = hc*bracket+field*(roottwohffield) # using hs=-hi then field is +ve not -ve, 
    
            hiflipminus=-hc*bracket+field*(roottwohffield) #see which way fields flips
            trialtemp=0

            if (temp < 0):  
                print('--------', i)            
                print('temp', temp, 'hi',hi*conv, 'hc', hc*conv, 'phitemp', phitemp)
                print('hiflipplus',hiflipplus*conv)
                print('hiflipminus',hiflipminus*conv)
                print('field',field, mu0*field*1e6)
                print('2*ht*phitemp',2*ht*phitemp*conv,phitemp,ht*conv)
                print('hc',hc*conv,beta)
                print('eglip', eglip, 'eglid', eglid, 'ht', ht)
                print('hf', hf, 'tm', tm, 'tau', tau, 'logtm_tau', log(tm/tau)) #ht = (roottwohffield)*hf*(log(tm/tau)) #new tm 
                print('aconst', aconst)
           
           
            if (hc >= (2*ht*phitemp)): #still loop over each hys, +1
        
                hcmore = hcmore + 1
   
                if ((hi > hiflipminus) and (hi < hiflipplus)): #+2 blocked
                  
                    memory = memory + 1
                    
                    if ((blockg[i] == 0) or (blockg[i] == 2) or (blockg[i] == -2)): #+3 prev blocked until this point
                        
                        if (hi >= (field*roottwohffield)): #+4
                            
                            blocktemp[i] = -1
                           
                        else:
                            #print('blockg = 0,2,-2 and hi < field*rootwo', 'blockg', blockg[i], 'hi', hi, 'field*roottwo', field*roottwohffield)
                            blocktemp[i] = 1 #end +3 unsure if sholud ended both or just one
                          
                    elif (blockg[i] == -1): #this line, already block stay , not need
                     
                        blocktemp[i] = -1
                

                    elif (blockg[i] == 1):
                        #print('blockg = 1', blockg[i], 'hi > hiflipminus and hi < hiflipplus', 'hi', hi, 'hiflipminus', hiflipminus, 'hiflipplus', hiflipplus)
                        blocktemp[i] = 1
                       
                    else:
                        #write to screen, blockg[i]
                        print(blockg[i], blocktemp[i]) #see if work - not need
                        print('----', i)
                    #end if +2, actually back to +2 if statement, still if hi
                elif (hi >= hiflipplus):#else field blocking above ht, this is hi test hiflip etc
                   
                    blocktemp[i] = -2
                  
                else:
                   
                    blocktemp[i] = 2
            else: #hc < 2*ht*phitemp. this is correctm meaning else above isnt
               
                hcless = hcless+1
        
                if ((hi < hiflipminus) and (hi > hiflipplus)): 
                   
                    blocktemp[i] = 0
                    counti = counti + 1 #what is this counting now, grains below intersect - SP? though aready find, lin 1644
                    #print('in new bounds so set blocktemp to zero, small front zone')
                else: #field blocking - below hc
                    if (hi >= hiflipminus):
                        #print('hi > hiflipminus (also greater than hiflipplus. hc < 2*ht*phitemp)', 'hc', hc, '2htphitemp', (2*ht*phitemp), 'hi', hi, 'hiflipminus', hiflipminus, 'hiflipplus', hiflipplus )
                        #print('field blocked big hi with small hc (blocktemp -2)', blocktemp[i])
                        blocktemp[i] = -2
                    else:
                        #print('hi <= hiflipplus and hi < hiflipminus', 'hi', hi, 'hiflipplus', hiflipplus, 'hiflipimus', hiflipminus)
                        #print('field blocked big negative hi with small hc (blocktemp +2)', blocktemp[i])
                        blocktemp[i] = 2

            if (temp < trialtemp):
                #need print to screen
                print('blocktemp', blocktemp[i])
               
        #end here - looked over each hys
        countiold = counti #number in hc
        
    totalm = 0.0
    totalmoment = 0
    totu = 0
    totb = 0
    i=0
    for i in range(num_hyss):
        #print('i for calc mom etc', i)
        blockg[i] = blocktemp[i]
        absblockg = abs(blockg[i])
        
        if (absblockg == 1): #if blockg 1 or -1
            if (boltz[i] < 0.00000001) and (boltz[i] > -0.000000001): #only zero
                #print('setting boltz as its 0 (check)', boltz[i])
                boltz[i] = tanh((field - histore[i])/hfstore[i])
           
                totalblock = totalblock + tanh((field - histore[i])/hfstore[i])
                totalblocktwo = totalblocktwo + tanh((field)/hfstore[i])
               
        if (blockg[i] == -2):
            #print('blockg = -2')
            moment = -0 #changed from zero to minues zero
        elif (blockg[i] == 2):
            #print('blockg = +2')
            moment = 0
        else:

            moment = blockg[i] #where does moment come from? - set to blockg if blockg not equal to 1

        #bit before is TRM acqution  
        #AF demag calc
        
        
        if (afswitch == 1): #afswtich given in function, 1 or 0
            afstore = V['afstore'] # move where relevant as no values unti demag - poss give dummy name to keep with rest of unpack of V
            #break
            #print('should not see this message, afswtich wrong')
            hi = histore[i]*(sqrt(2))
            hc = hcstore[i]*(sqrt(2))
            hc=hc*(((sin(hys[i,6])**(2./3.))+(cos(hys[i,6])**(2./3.)))**(-3./2.)) #hc=hc*(((sin(hys[i,5])**(2./3.))+(cos(hys[i,5])**(2./3.)))**(-3./2.))
    
            af = afstore/(1000*mu0) #define later. afstore number
            if (hi > 0) and (hi > (hc-af)): #whats this bit mean
                moment = 0
                blockg[i] = -2
                totu = totu -1 #total unblocked, decrease 
               
            if (hi < 0) and (hi < (-hc + af)):
                moment = 0
                blockg[i] = 2
            
                totb = totb +1 #total blocked increase?, just diff ways? is this not the sme outcome
                boltz[i] = 1.0
        totalm = totalm + abs(moment)*abs(cos(hys[i,5]))*hys[i,3]*beta*sense[i]*(boltz[i]) #add in forcdist (hys[i,3])

        aftotalm = totalm #unsure if this is true
        sir = totalm #saying all totalm
        if (beta < 0):
            print('print beta < 0')
        #if (boltz[i] < 0):
         #   print('boltz < 0')
        if ( hfstore[i] < 0):
            print('hf <0')

        blockper=blockper+abs(moment)*1.0 

        totalmoment=totalmoment+moment
        
    blockper=blockper/(1.0*num_hyss)

    V['aftotalm'] = aftotalm
    V['sir'] = sir
    V['counttime'] = counttime
    V['blockper'] = blockper
    V['countiold'] = countiold 
    V['blocktemp'] = blocktemp
    V['boltz'] = boltz
    V['blockg'] = blockg
    V['totalm'] = totalm
    V['totalblocktwo'] = totalblocktwo
    V['totalblock'] = totalblock
    V['totalmoment'] = totalmoment
    V['tm'] = tm
    V['sense'] = sense

    return #totalm
    #end of blocktemp function - inputs etc - global        
  
def sirm_test(V, X):
    sirm = V['sirm']
    cntfield = X['cntfield']
    name = X['name']
    ifield = V['ifield']
    w, h = figaspect(1) #from https://stackoverflow.com/questions/48190628/aspect-ratio-of-a-plot
    fig, ax = plt.subplots(figsize=(w,h))
    demagstep = X['af_step']
    #set first value as v v low just so remove zeros to plot easily
    demagstep2 = demagstep
    demagstep2[0] = 0.0001

    demagstep2 = demagstep2[demagstep2 != 0]

    sirm2 = sirm
    sirmn = np.copy(sirm2)

    for i in range(4):
        for j in range(23): #change back to 23 
            sirmn[i][j] = (sirm2[i][j]/sirm2[i][0])


    af_sirm_n = X['af_irm']

    norm = af_sirm_n[0]

    af_sirm_n_n = np.copy(af_sirm_n)
    for i in range(len(af_sirm_n)):
        af_sirm_n_n[i] = af_sirm_n[i]/norm

    sirm_p = sirmn[:ifield,:cntfield] #change 4 to no fields

    plt.plot(demagstep2, sirm_p[2], marker = 'o', color = 'r') #x = 22, y = 100

    plt.plot(demagstep2, af_sirm_n_n, marker = 'o', color = 'b')        
    plt.ylabel('SIRM (normalized %)')
    plt.xlabel('af demag step')
    plt.title("Demagnetisation of the SIRM for sample '{0}'".format(name))

    plt.text(30, 0.9, r'measured ', fontsize=12) #poss change to percentage across line (max/50 etc)
    plt.plot(25,0.9, marker = 'o', color='b')
    plt.text(30, 0.8, r'simulated', fontsize=12)
    plt.plot(25, 0.8, marker = 'o', color='r')
    plt.savefig('sirm_test_web.pdf')
    plt.show
    return
    
def calc_pal(X, V):
    cntfield = X['cntfield']
    ifield = V['ifield']
    afmag = V['afmag']
    sirm = V['sirm'] #sirm seems global so shoulld have worked anwyay
    af_nrm_n = X['af_nrm']
    af_sirm_n = X['af_irm']
    name = X['name']
    aconst = V['aconst']
    fields = V['fields']
    af_step = X['af_step']
    tempmin = V['tempmin']
    tempmax = V['tempmax']
    afnegsirm = 0.0         
    averageme = 0.0
    averageunweight = 0.0
    averagemeadj = 0.0
    averagecount = 0
    flatcount = np.zeros((10))
    noflatsub = 0
    averageflat = np.zeros((10))
    trmsirmratio = 0.0
    sigmtot = 0.0
    selrms = 0.0
    sumegli = 0

    ###################
    afpick = 6 #pick from user - Z-plot

    while True:
            afpick = (input("Pick the AF demag step where the primary componant is identified (tutorial) (~6):" ))
            print(type(afpick))
            try:
                afpick = int(afpick)
                if (afpick >= 0) and (afpick <= cntfield): #within range of AF demag steps - may break if pick too high
                    print('in bounds')
                    break
            except ValueError:
                print('Not an interger')
                True
           # if (isinstance(sf_choose, int)):
                print('int')

    #######################
    sit = ifield
    std = np.zeros((100))
    ys = np.zeros((100))
    used = np.zeros((50))
    flatused = np.zeros((50))
    ystore = np.zeros((100))
    sigmstore = np.zeros((100))
    flatstore = np.zeros((10,100))
    flatdi = np.zeros((10))
    sigmflat = np.zeros((10))
    flatmedianme = np.zeros((10))
    sigmtotflat = np.zeros((10))
    cntfieldaf = cntfield
    actfield = 1 #random number but need to see what this is
    dispersion = 0
    vtwo = 0
    fortyone = 0

    shortdivlist = []
    shortminuslist = []
    ilist = []
    af_step_list = []


    flat = 0 #set here as well as earlier to save time when testing
    afratio = 0
    #do the shortdiv etc seperate

    for i in range(cntfieldaf): #cntfieldsaf

        xlist = []
        ylist = []
        sumx = 0.
        sumy = 0.
        sumxy = 0.
        sumxx = 0.

        for j in range(sit): #for each field - a
            
            
            y = fields[j]*mu0*1e6 #field time something, x, y ust values
            x = afmag[j,i]/sirm[j,i] #norm af mag to sirm - sirm after that stage of demagetisation
            #plotting points
            xlist.append(x)
            ylist.append(y)

            if (sirm[j,i] == 0):

                for i in range(afpick, averagecount): #this all moves to diff point
                    #afpick given by file called sirmms.dat
                    dispersion = dispersion + (((ystore[i] - averageme/sigmtot))**2)/sigmstore[i]
                    vtwo = vtwo + (1/sigmstore[i])**2
                    fortyone = 1

            sumx = sumx + x        
            sumxx = sumxx + x**2
            sumxy = sumxy + x*y
            sumy = sumy + y

        mfit=(((sit+000.0)*sumxy)-sumx*sumy)/(((sit+000.0)*sumxx)-sumx*sumx)
        cfit=(sumy*sumxx-sumx*sumxy)/((sit+000.0)*sumxx-sumx*sumx)

        xtest = np.linspace(min(xlist), max(xlist), 10)
        #print('xtest', xtest)
        xtest = np.array(xtest)
        ytest = np.copy(xtest)
        ytest = mfit*xtest + cfit
        sumdi = 0.
        xlist2 = []
        ylist2 = []
        dilist = []
        for j in range(sit): #x and y values set for each field again
            y= fields[j]*mu0*1e6
            x=(afmag[j,i]/sirm[j,i]) #same as above, line 1024
            di = y - (mfit*x+cfit) #see how closely x and y fit with line equation
            dilist.append(di)
            sumdi = sumdi + di**2
            xlist.append(x)
            ylist.append(y)
        sumdi = sumdi/(sit-2) 
        sigm = sit*sumdi/(sit*sumxx-sumx*sumx) #variance
        x = af_nrm_n[i]/af_sirm_n[i]   
        y = x*mfit +cfit #field assoacted witht his step for demag  
        xlist2.append(x)
        ylist2.append(y)

        ys[i] = y

        std[i] = sqrt(sigm)*x
        used[i] = 0 #set used for each cntfield to zero
        flatused[i] = 0
        
        #here
        shortdiv=abs(1-((sirm[j-1,i]/sirm[j-1,0])/(af_sirm_n[i]/af_sirm_n[0])))*100 #do for 3rd field
        shortminus=abs(((sirm[j-1,i]/sirm[j-1,0])-(af_sirm_n[i]/af_sirm_n[0])))*100
        shortdivm=abs(1-((sirm[j-2,i]/sirm[j-2,0])/(af_sirm_n[i-1]/af_sirm_n[0])))*100
        shortminusm=abs(((sirm[j-2,i]/sirm[j-2,0])-(af_sirm_n[i-1]/af_sirm_n[0])))*100
        shortdivlist.append(shortdiv)
        shortminuslist.append(shortminus)
        af_step_list.append(af_step[i])
        if (i >= afpick) and (sumx != 0.0): 

            sigm = sigm*(x**2)


            afratio = afmag[2,i]/afmag[2,0]

            if (i >= 1): #greater than 1 as 0 counts here
                afratiom = (afmag[2, i-1]/afmag[2,0]) #also do ratio for prev point - poss compares afratio and afratiom

            shortdiv=abs(1-((sirm[j-1,i]/sirm[j-1,0])/(af_sirm_n[i]/af_sirm_n[0])))*100
            shortminus=abs(((sirm[j-1,i]/sirm[j-1,0])-(af_sirm_n[i]/af_sirm_n[0])))*100
            shortdivm=abs(1-((sirm[j-2,i]/sirm[j-2,0])/(af_sirm_n[i-1]/af_sirm_n[0])))*100
            shortminusm=abs(((sirm[j-2,i]/sirm[j-2,0])-(af_sirm_n[i-1]/af_sirm_n[0])))*100
            sigm=abs(1-sigm)

            #shortdivlist.append(shortdiv)
            #shortminuslist.append(shortminus)
            ilist.append(i)
            #af_step_list.append(af_step[i])
            if (i > 0) and (i < 30): #30 seems arbituary

                if (af_nrm_n[i]/af_nrm_n[0] > 0.01) and (afratio > 0.01): #stage requires af ratio been given value but not needed

                    if (y > 0.0):
                        #print('y>0', y)

                        if (shortdiv < 100): #acceptable ranges of S - seen in data displayed
                            if (shortminus < 20):
                                print('conds are met')
                                selrms=selrms+abs((af_sirm_n[i]/af_sirm_n[0])- (sirm[j-1,i]/sirm[j-1,0]))               

                                sigmtot=sigmtot+(1/sigm) # sum of 1/variance

                                sumegli=sumegli+(y/actfield) #sum of fields used in predied

                                averageme=averageme+(y)/sigm 

                                averageunweight=averageunweight+y

                                averagecount=averagecount+1

                                ystore[averagecount]=y #not want averagecount to skip points?

                                sigmstore[averagecount]=sigm

                                used[i]=1 #array, this means use tihs point in calc? therefore add to used array
                                trmsirmratio=trmsirmratio+x #x is true trm/sirm measred ratio

                                if (i > 1):
                                    flatdiff = abs(y - ys[i-1])/max(y,ys[i-1])

                                    if (flatdiff < 0.2) or (y < 3.0 and ys[i-1] < 3.0): #if flatdiff < 0.2 or both <3

                                        if (noflatsub == 0): # move onto new section

                                            flat = flat +1

                                            if (i-1 < afpick) or (shortdivm > 100) or (shortminusm > 20) or (ys[i-1] < 0.0) or (af_nrm_n[i-1]/af_nrm_n[0] < 0.01) or (afratiom < 0.01):
                                                print('rejecting prev point as out of bounds or not close enough, uses lookin 2 points back?')
                                            else:

                                                flatcount[flat] = flatcount[flat] + 1

                                                flatstore[flat][int(flatcount[flat])] = ys[i-1] #index erroe - change to int

                                                averageflat[flat] = averageflat[flat] + ys[i-1] #page 1121

                                                sigmflat[flat] = sigmflat[flat] + (1/sigm) #sum of 1/variance

                                                flatused[i-1] = flat

                                                flatdi[flat] = flatdi[flat]+(y-ys[i-1])/max(y,ys[i-1])

                                        noflatsub = 1

                                        flatcount[flat] = flatcount[flat] + 1

                                        flatstore[flat][int(flatcount[flat])] = y

                                        averageflat[flat] = averageflat[flat] + y #add one this point to list, some overwrite what above some diff
                                        sigmflat[flat] = sigmflat[flat] + (1/sigm) # sum of 1/variance
                                        flatdi[flat] = flatdi[flat] + (y - ys[i-1])/max(y,ys[i-1])
                                        flatused[i] = flat
                                    else: #one indent because one end if ending the else which not needed in python

                                        noflatsub = 0

                                else:
                                    noflatsub = 0

                            else:
                                noflatsub = 0


                        else:
                            noflatsub = 0

                    else:
                        noflatsub =0

                else:
                    noflatsub = 0


            else:
                noflatsub = 0

        else: 
            noflatsub = 0
    if (averagecount != 0):
        selrms=100*(selrms)/(1.0*averagecount) #line 1179

    #calc median
    temptwo = np.empty((100))
    temptwo[:] = np.nan

    for i in range(averagecount):
        temptwo[i] = ystore[i]

    temptwo = temptwo[~np.isnan(temptwo)] #want to avoid extra zeros
    medianme = np.median(temptwo)

    for jf in range(flat):

        flatc = 0

        for i in range(int(flatcount[jf])): #line 1207 - called calc flat median

            if (i >= 1): #multiple points to ave over - shoudl it be greater than just 2?

                if (flatstore[jf,i] > 1.0*flatstore[jf,i-1]): #unsure if shold be i-1 or i
                    flatc = flatc +1

                if (flatstore[jf,i] < 1.0*flatstore[jf,i-1]): #looking at point behind it
                    flatc = flatc - 1


        if (abs(flatc) == (flatcount[jf] -1)) and (flatcount[jf] > 3): #unsure if should be greater than 2 or greater than 3

            if (abs(flatstore[jf,0] - flatstore[jf,flatcount[jf]]) > 10):
                flatcount[jf] = 0
                #go to 911

                break #unsure if this is correct
        else:
            print('conds not met') #1st point do this as zero

        if (flatcount[jf] <= 1): #assumes just needs 2 points included

            flatcount[jf] = 0
            #go to 911
            break #cut of whole loop - cut of loop for jf in range flat - seems correct

        #calc flat median #for i in range(flatcount[jf] - 1): #unsure if this is the correct limit
        flatmedianme[jf] = np.median(temptwo[:, jf])   #temptwo[i] = flatsotre range for i in length flatcount add in values , try just do upto certain amount


    #set dispersion etc - line 1257
    if (fortyone == 0):
        dispersion = 0.0
        vtwo = 0.0 #but not if reached 41 - in 41 section - set variable to one 

    for i in range(afpick, averagecount): #this all moves to diff point
        #afpick given by file called sirmms.dat
        dispersion = dispersion + (((ystore[i] - averageme/sigmtot))**2)/sigmstore[i]
        vtwo = vtwo + (1/sigmstore[i])**2


    #flat section normal districution standard distribution
    for i in range(flat):
        sigmtotflat[i] = 0
        for k in range(int(flatcount[flat])):

            sigmtotflat[i] = sigmtotflat[i] + ((flatstore[i,k] - (averageflat[i]/flatcount[i]))**2)/flatcount[i]

        sigmtotflat[i] = sqrt(sigmtotflat[i])
        #end of flat normal

    sirmrms = 0
    print("All demag and palaeointensity data is saved in afdemag-all-{0}.dat".format(name))
    print("All demag and data regarding the SIRM data is saved in afdemag-sirm-{0}.dat".format(name))
    fall = open("afdemag-all-{0}.dat".format(name), "w") #'w' will overwrite file is already exists
    fall.write('afield' + '\t' + 'step' + '\t' + 'stepPI' + '\t' + 'std' + '\t' + 'select' + '\t' + 'flatno'+ '\t' + 'shortminus' + '\t' + 'SIRMAFS%' + '\t' + 'SIRMAFM%' + '\t' + 'AFNRM/SIRM-M%' + '\t' + 'AFNRM/SIRM-S%' + '\t' +  'AFNRM/NRM-M%' + '\t' + 'AFNRM/NRM-S%' + 'shortdiv%')
    fsirm = open("afdemag-sirm-{0}.dat".format(name), "w")

    fsirm.write('afield' + '\t' + 'measured' + '\t' + 'simulated')
    for i in range(cntfieldaf): #0 to 21
        sirmrms = sirmrms + abs((af_sirm_n[i]/af_sirm_n[0]) - sirm[j-1, i]/sirm[j-1][0])
        #print(sirmrms) #increases 0.004 -> 2.6
        fall.write('\n')
        #f.write(str(af_step[i]))
        fall.write(str(af_step[i]) + '\t' + str(i) + '\t' + str(ys[i]) + '\t' + str(std[i]) + '\t' + str(used[i]) + '\t' + str(flatused[i]) + '\t' + str((((sirm[j-1,i]/sirm[j-1,0])-(af_sirm_n[i]/af_sirm_n[0])))*100) + '\t' + str(sirm[j-1,i]/sirm[j-1,0]*100) + '\t' + str((af_sirm_n[i]/af_sirm_n[0])*100) + '\t' + str((af_nrm_n[i]/af_sirm_n[i])*100) + '\t' + str((afmag[j-1,i]/sirm[j-1,i])*100) + '\t' + str((af_nrm_n[i]/af_nrm_n[0])*100) + '\t' + str((afmag[j-1,i]/afmag[j-1,0])*100) + '\t' + str(abs(1-((sirm[j-1,i]/sirm[j-1,0])/(af_sirm_n[i]/af_sirm_n[0])))*100))
        fsirm.write('\n')
        fsirm.write(str(af_step[i]) + '\t' + str(af_sirm_n[i]/af_sirm_n[0]) + '\t' + str(sirm[j-1, i]/sirm[j-1,0]))

    fsirm.close()

    fall.write('\n')

    sirmrms = sirmrms/cntfieldaf

    fall.write('SIRM MEAN DIFFERENCE % =' + '\t '+ str(100*sirmrms))
    fall.write('\n')
    if (averagecount !=0) and (averagecount !=1) and (sigmtot !=0): #only do both is average at least 1 
        sampleunbias=dispersion*(sigmtot/((sigmtot**2)-vtwo))
        dispersion=dispersion/(averagecount-1)


    if (averagecount == 1): #here equals 11
        dispersion = 1

    if (sigmtot == 0):
        sigmtot = 1
        
        print('sigm tot = 0 weighted average not possible')
        
    if (averagecount == 0):
        averagecount = 1
        print('avercount = 0 weighted average not possible')   
        
    fall.write('weighted average = ' + '\t' + str(averageme/(sigmtot)) + '\t' + str(sqrt(dispersion/sigmtot)))
    fall.write('\n')
    fall.write('unweighted average = ' + '\t' + str(averageunweight/averagecount))
    fall.write('\n')
    fall.write('unweighted median = ' + '\t' + str(medianme))
    fall.write('\n')
    fall.write('cntfields = (cntfield, cntfieldaf, afpick) ' + '\t' + str(cntfield) + '\t' + str(cntfieldaf) + '\t' + str(afpick))


    #determining which is best flat 
    maxjf = 0
    minsig = 10000
    jfs = 0

    for jf in range(flat):

        if ((flatcount[jf] >= maxjf) and (flatcount[jf] > 0)):

            if ((flatcount[jf] == maxjf) or (maxjf == 1)): #change maxjf == 0

                if (sigmtotflat[jf] < minsig): #seem to do same thing regardnless of these if statements

                    minsig = sigmtotflat[jf]
                    maxjf = flatcount[jf]
                    jfs = jf

            else:
                minsig = sigmtotflat[jf]
                maxjf = flatcount[jf]
                jfs = jf


        if (flatcount[jf] > 0):

            fall.write('\n')
            fall.write('unweighted average flat (jf, averageflat(jf)/flatcount(jf), sigmtotflat(jf, flatdi(jf)) =' + '\t' + str(jf) + '\t' + str(averageflat[jf]/flatcount[jf]) + '\t' + str(sigmtotflat[jf]) + '\t' + str(flatdi[jf]) + '\t' + str(flatcount[jf]) + '\t' + str(flatdi[jf]/flatcount[jf]))
            fall.write('\n')
            fall.write('unweighted flat median (jf, flatmedianme(jf) = ' + '\t' + str(jf) + '\t' + str(flatmedianme[jf]))
        else:

            fall.write('no points selected for flat section (jf)' + '\t' + str(jf))


    fall.close()        
    aconst = -aconst*log(0.01*(tempmin)/(tempmax-tempmin))/3600
    print('Output results to averageout_{0}.dat'.format(name))
    fave = open('averageout_{}.dat'.format(name), 'w') #117 file
    if (averageme > 0):

        if (jfs != 0):

            fave.write(str(averageme/(sigmtot)) + '\t' + str(sqrt(dispersion/sigmtot)) + '\t' + str(averagecount) + '\t' + str(averageunweight/averagecount) + '\t' + str(medianme) + '\t' + str(flatcount[jfs]) + '\t' +  str(averageflat[jfs]/(1.0*flatcount[jfs])) + '\t '+  str(sigmtotflat[jfs]) + '\t' + str(flatmedianme[jfs]) + '\t' + str(aconst) + '\t' + '147' + '\t' + str(100*af_nrm_n[0]/af_sirm_n[0]) + '\t' + str(sirmrms) + '\t' + str(selrms))
        else:
            fave.write(str(averageme/sigmtot) + '\t' + str(sqrt(dispersion/sigmtot)) + '\t' + str(averagecount) + '\t' + str(averageunweight/averagecount) + '\t' + str(medianme) + '\t' + '0.0' + '\t' + '0.0' + '\t' + '0.0' + '\t' + str(aconst) + '\t' + '147' + str(100*(af_nrm_n[0]/af_sirm_n[0])) + '\t' + str(sirmrms) + '\t' + str(selrms))
    else:
        fave.write('no points selected' + '\t' + str(sirmrms))


    fave.close()
    V['shortdivlist'] = shortdivlist
    V['shortminuslist'] = shortminuslist
    V['ys'] = ys
    
    return(X,V)
    
    
def plot_sirm_check(X,V):
    af_step_list = X['af_step']
    shortdivlist = V['shortdivlist']
    shortminuslist = V['shortminuslist']
    name = X['name']
    shortdivlistplot = np.array(shortdivlist[:(len(af_step_list))])
    shortminuslistplot = np.array(shortminuslist[:(len(af_step_list))])
    shortdivlistplot[np.isnan(shortdivlistplot)] = 0
    shortminuslistplot[np.isnan(shortminuslistplot)] = 0
    twenty = []
    hundred = []

    for i in range(len(af_step_list)):
        twenty.append(20)
        hundred.append(100)
    w, h = figaspect(1) #from https://stackoverflow.com/questions/48190628/aspect-ratio-of-a-plot
    fig, ax = plt.subplots(figsize=(w,h))

    plt.plot(af_step_list, twenty, 'b') #af step list is just af_step?
    plt.plot(af_step_list, hundred, 'r')
    plt.plot(af_step_list, shortdivlistplot,  marker='o', color= 'r')
    plt.plot(af_step_list, shortminuslistplot,  marker='o', color = 'b')
    plt.title('SIRM checks looking at difference between measured and simulated. Sample {0}'.format(name))

    plt.xlabel('AF peak (mT)')
    plt.ylabel('S_diff or S_ratio (%)')
    plt.text(25, 85, r'S diff ', fontsize=12)
    plt.plot(22,87, marker = 'o', color='b')
    plt.text(25, 75, r'S ratio', fontsize=12)
    plt.plot(22, 77, marker = 'o', color='r')
    print('Figure saved as SIRM-checks-{0}.pdf'.format(name))
    plt.savefig('SIRM-checks-{0}.pdf'.format(name))
    plt.show
    return
    
def plot_pal(V,X):
    
    ys = V['ys']
    af_step = X['af_step']
    
    w, h = figaspect(1) #from https://stackoverflow.com/questions/48190628/aspect-ratio-of-a-plot
    fig, ax = plt.subplots(figsize=(w,h))

    #plot with red colour and calc average
    #selected_mean = np.mean(ys[8:20])
    #mean_dev = np.std(ys[8:20])
    #selected_med = np.median(ys[8:20])
    plt.plot(af_step,ys[:len(af_step)], 'b')
    plt.plot(af_step,ys[:len(af_step)],  marker='o', color= 'b')
    #plt.plot(af_step[8:20],ys[8:20],  marker='o', color= 'r')
    
    #plt.text(20, 6, r'selected median: %.2f $\mu$T'%selected_med, fontsize=11)
    #plt.text(20, 7, r'rejected mean: %.2f $\pm$ %.2f $\mu$T'%(selected_mean ,mean_dev), fontsize=11)
    #plt.text(20, 6, r'selected median: %.2f $\mu$ T'%selected_med, fontsize=12)
    plt.xlabel('AF peak (mT)')
    plt.ylabel('paleointensity (\u03BCT)')

    plt.title('TRM PI est (Me)')

    #plt.savefig('ESA_MF-PI_ESTS_colour.pdf')
    plt.show
    
    ind = []
    for i in range(len(af_step)):
        ind.append(i)
        i+=1


    ind = np.array(ind)

    print("AF step index = AF step")
    for n, v in zip(ind, af_step):
        print("{} = {}".format(n, v))
    
def fin_pal(X,V):
    ys = V['ys']
    af_step = X['af_step']
    name = X['name']
    cntfield = X['cntfield']
    
    while True:
            low_b = (input("Pick the AF step number for the lower bound of the platau palaeointensity region to calcualte the median palaeointensity from:"))
            #print(type(afpick))
            try:
                low_b = int(low_b)
                if (low_b >= 0) and (low_b <= cntfield): #within range of AF demag steps - may break if pick too high
                    print('in bounds')
                    break
            except ValueError:
                print('Not an interger')
                True
           # if (isinstance(sf_choose, int)):
                print('int')
            
    while True:
            up_b = (input("Pick the AF step number for the upper bound of the platau palaeointensity region to calcualte the median palaeointensity from:"))
            
            try:
                up_b = int(up_b)
                if (up_b >= low_b) and (up_b <= cntfield): #within range of AF demag steps - may break if pick too high
                    
                    break
                else:
                    print('in bounds, must be above the lower bound')
            except ValueError:
                print('Not an interger')
                True
           # if (isinstance(sf_choose, int)):
                print('int')
            
            #plot restulting  same graoh with labelled in red and wiht median etc 
            
    ys = V['ys']
    af_step = X['af_step']
    
    w, h = figaspect(1) #from https://stackoverflow.com/questions/48190628/aspect-ratio-of-a-plot
    fig, ax = plt.subplots(figsize=(w,h))

    #plot with red colour and calc average
    selected_mean = np.mean(ys[low_b:up_b+1]) #check these include these values
    mean_dev = np.std(ys[low_b:up_b+1])
    selected_med = np.median(ys[low_b:up_b+1])
    plt.plot(af_step,ys[:len(af_step)], 'b')
    plt.plot(af_step,ys[:len(af_step)],  marker='o', color= 'b')
    plt.plot(af_step[low_b:up_b+1],ys[low_b:up_b+1],  marker='o', color= 'r')
    
    plt.text(max(af_step)/4., max(ys[:len(af_step)])/1.2, r'selected median: %.2f $\mu$T'%selected_med, fontsize=11)
    plt.text(max(af_step)/4., max(ys[:len(af_step)])/1.4, r'rejected mean: %.2f $\pm$ %.2f $\mu$T'%(selected_mean ,mean_dev), fontsize=11)
    
    plt.xlabel('AF peak (mT)')
    plt.ylabel('paleointensity (\u03BCT)')

    plt.title('TRM PI est (Me)')
    print('Figure saved as PI_est_{0}.pdf'.format(name))
    plt.savefig('PI_est_{0}.pdf'.format(name))
    plt.show
    #add to file
    return


