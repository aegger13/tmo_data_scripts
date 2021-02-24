import numpy as np
import psana as ps
import sys
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from scipy.ndimage.filters import gaussian_filter

exp = 'tmoc00118' #
run_number = int(sys.argv[1])
preprocessed_folder = '/reg/d/psdm/tmo/%s/scratch/preproc/v1/' % exp
filename = preprocessed_folder+'run%d_v1.h5' % run_number
if (os.path.isfile(filename) or os.path.isfile(filename.split('.')[0]+'_part0.h5')):
    raise ValueError('h5 files for run %d already exist! check folder: %s'%(run_number, preprocessed_folder))
    #not sure which one to check for so check for both

ds = ps.DataSource(exp=exp, run=run_number)
smd = ds.smalldata(filename=filename)

update = 50 # Print update (per core)
default_val = -9999.0

######### ion ToF settings ###############
#m = 44.0 # parent mass
#c, t0 = 1.218735268290037, 0.204053663825002 # ToF calibration
#def ToF(m_q, c, t0):
#    return t0 + c * np.sqrt(m_q)
#qs = np.concatenate((np.arange(1, 9), [50, 0.5])) # qs to look at 
#ts = ToF(m / qs, c, t0)
#rebin_factor = 10
##########################################

# ######### KToF preproc settings ##########
# ktofslice_t1, ktofslice_t2= 0.2,1.1 #upper and lower bounds in us
# nkeep_ktof = 2000 # max number of hits to save
# ##########################################

######### 1d hit finder ##################
cfd_folder = '/reg/d/psdm/tmo/tmolw0618/results/modules/'
sys.path.append(cfd_folder)
from cfd import cfd
shift_cfd = 2e-3 # in microseconds
deadtime_cfd = 2e-3 # in microseconds
threshold_cfd = 2
sig_cfd = 4
##########################################

######### 2d hit finder ##################
hitfinder_folder = '/reg/neh/home/tdd14/modules/hitfinder_py3/'
sys.path.append(hitfinder_folder)
from hitFinder_peakvals import findHits, make_kernel

nkeep_eVMI = 2000 ## max number of hits to save

thresh_eVMI = 38.
mkr_eVMI = 1
gkr_eVMI = 5
sigma_eVMI = 0.9

gfilt_eVMI = make_kernel(gkr_eVMI, sigma_eVMI)
##########################################
Nfound = 0
Nbad = 0
# times = None #removed 20210215
timesktof = None

for run in ds.runs():
    
    timing = run.Detector("timing")
#     hsd = run.Detector("hsd")
    tmo_opal1 = run.Detector("tmo_opal1") #looking at c-VMI
    tmo_opal2 = run.Detector("tmo_opal2") #looking at microscope
    tmo_atmopal = run.Detector("tmo_atmopal")
    andor = run.Detector("andor")
    gmd = run.Detector("gmd")
    xgmd = run.Detector("xgmd")
    ebeam = run.Detector("ebeam")
    
    if hasattr(run, 'epicsinfo'):
        epics_strs = [item[0] for item in run.epicsinfo.keys()][1:] # first one is weird
        epics_detectors = [run.Detector(item) for item in epics_strs]

    for nevent, event in enumerate(run.events()):
        
        if nevent%update==0: print("Event number: %d, Valid shots: %d" % (nevent, Nfound))
            
        data = {'epics_'+epic_str: epic_det(event) for epic_str, epic_det in zip(epics_strs, epics_detectors)}
        
        if any(type(val) not in [int, float] for val in data.values()):
            print("Bad EPICS: %d" % nevent)
            Nbad += 1
            continue
            
#         temporarily out of DAQ         
#         hsd_data = hsd.raw.waveforms(event)
#         if hsd_data is None:
#             print("Bad HSD: %d" % nevent)
#             Nbad += 1
#             continue
        im = tmo_opal1.raw.image(event)
        if im is None:
            print("Bad OPAL1: %d" % nevent)
            Nbad += 1
            continue    
        im2 = tmo_opal2.raw.image(event)
        if im2 is None:
            print("Bad OPAL2: %d" % nevent)
            Nbad += 1
            continue
#         imatm = tmo_atmopal.raw.image(event)
#         if imatm is None:
#             print("Bad ATM OPAL: %d" % nevent)
#             Nbad += 1
#             continue
        vls = andor.raw.value(event)
        if vls is None:
            print("Bad VLS: %d" % nevent)
            Nbad += 1
            continue
        evrs = timing.raw.eventcodes(event)
        if evrs is None:
            print("Bad EVRs: %d" % nevent)
            Nbad += 1
            continue
        evrs = np.array(evrs)
        if evrs.dtype == int:
            data['evrs'] = evrs.copy()
        else:
            print("Bad EVRs: %d" % nevent)
        
        bad = False
        for (detname, method), attribs in run.detinfo.items():
            if bad: break
            if (detname not in ['timing', 'hsd', 'tmo_opal1', 'tmo_opal2', 'andor', 'epicsinfo']) and not (detname=='tmo_atmopal' and method=='raw'):
                for attrib in attribs:
                    if detname!='tmo_atmopal' or attrib not in ['proj_ref', 'proj_sig']:
                        val = getattr(getattr(locals()[detname], method), attrib)(event)
                        if val is None:
                            if detname in ['ebeam', 'gmd', 'xgmd'] and evrs[161]: # BYKIK gives None for these, but we still want to process the shots
                                val = default_val
                            else:
                                bad = True
                                print("Bad %s: %d" % (detname, nevent))
                                Nbad += 1
                                break
                        data[detname+'_'+attrib] = val
        if bad:
            continue
        
#         if times is None:
#             times = hsd_data[0]['times'] * 1e6
#             #ktofslice_idx=np.argmin(abs(times-ktofslice_ts[0])), np.argmin(abs(times-ktofslice_ts[1])) # times to indices
#         # get KTof waveform data
#         if timesktof is None:
#             timesktof = hsd_data[6]['times'] * 1e6
#         wf = hsd_data[0][0].astype('float')
        
#         wf_diode = hsd_data[9][0].astype('float')
#         for i in range(4):
#             wf[i::4] -= wf[i:250:4].mean(0)
#             wf_diode[i::4] -= wf_diode[i:250:4].mean(0)
            
#         data['iToF_wf'] = wf.reshape(-1, rebin_factor).mean(1)
#         data['diode_wf'] = wf_diode[:5000].copy()
        
        # get eVMI hits
        xc_, yc_, pv_ = findHits(im.flatten(), gfilt_eVMI, thresh_eVMI, gkr_eVMI, mkr_eVMI)
        
        if len(xc_) <= nkeep_eVMI:
            xc, yc, pv = np.ones(nkeep_eVMI)*default_val, np.ones(nkeep_eVMI)*default_val, np.ones(nkeep_eVMI)*default_val
            xc[:len(xc_)] = xc_
            yc[:len(xc_)] = yc_
            pv[:len(xc_)] = pv_
        else:
            xc, yc, pv = np.array(xc_[:nkeep_eVMI]), np.array(yc_[:nkeep_eVMI]), np.array(pv_[:nkeep_eVMI])
            print("warning: eVMI too long!")
            
        data['xc'] = xc.copy()
        data['yc'] = yc.copy()
        data['pv'] = pv.copy()
        
        #VLS:
        vls = vls.mean(0)
        data['vls'] = vls.copy()
            
         # now add KToF processing
#         wfktof = hsd_data[6][0].astype('float') #6 is KToF channel
#         for i2 in range(4):
#             wfktof[i2::4] -= wfktof[-5000+i2::4].mean(0)
#         #ktofslice=wfktof[ktofslice_idx[0]:ktofslice_idx[1]+1]
#         ktofslice = wfktof[(ktofslice_t1<timesktof)&(timesktof<ktofslice_t2)]
        
#         data['ktofslice'] = ktofslice.copy()
        
        # now run cfd on KToF
#         tpk_, Ipk_ = cfd(times, gaussian_filter(-wfktof, sig_cfd), shift_cfd, threshold_cfd, deadtime_cfd)
        
#         if len(tpk_) > nkeep_ktof:
#             tpk, Ipk = tpk_[:nkeep_ktof], Ipk_[:nkeep_ktof]
#             print("warning: KToF too long!")
#         else:
#             tpk, Ipk = np.ones(nkeep_ktof)*default_val, np.ones(nkeep_ktof)*default_val
#             tpk[:len(tpk_)] = tpk_
#             Ipk[:len(Ipk_)] = Ipk_
            
#         data['ktofpk'] = tpk.copy()
#         data['ktofIpk'] = Ipk.copy()
        
        valid_data = True
        for key, val in data.items():
            if (type(val) not in [int, float]) and (not hasattr(val, 'dtype')):
                print("Bad data:", key)
                valid_data = False
                break
        
        if valid_data:
            smd.event(event, **data)
            Nfound += 1
        else:
            Nbad += 1
            continue
        
if smd.summary:
    Nbad = smd.sum(Nbad)
    Nfound = smd.sum(Nfound)
    smd.save_summary(Nfound=Nfound, Nbad=Nbad, thresh_eVMI=thresh_eVMI, mkr_eVMI=mkr_eVMI, gkr_eVMI=gkr_eVMI, sigma_eVMI=sigma_eVMI, shift_cfd=shift_cfd, threshold_cfd=threshold_cfd, deadtime_cfd=deadtime_cfd, sig_cfd=sig_cfd)
    
#     smd.save_summary(Nfound=Nfound, Nbad=Nbad, rebin_factor=rebin_factor, meanktofslice_t1=ktofslice_t1, ktofslice_t2=ktofslice_t2, thresh_eVMI=thresh_eVMI, mkr_eVMI=mkr_eVMI, gkr_eVMI=gkr_eVMI, sigma_eVMI=sigma_eVMI, shift_cfd=shift_cfd, threshold_cfd=threshold_cfd, deadtime_cfd=deadtime_cfd, sig_cfd=sig_cfd)
    
smd.done()
    
#if rank == (size - 1):
#    perms = '444' # fo-fo-fo
#    for f in [filename.split('.')[0]+'_part0.h5', filename]:
#        os.chmod(f, int(perms, base=8))
