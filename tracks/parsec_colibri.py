import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from ..utils import replace_, get_zy
from .track import Track, AGBTrack
from ResolvedStellarPops.fileio import get_files, get_dirs, ensure_dir
from ..config import *

class FirstTP(object):
    """
    Input files between PARSEC and COLIBRI
    """
    def __init__(self, filename):
        self.load_inp(filename)
        self.base, self.name = os.path.split(filename)
        self.Z, self.Y = get_zy(filename)

    def load_inp(self, filename):
        rdict = {'#': '', 'm1': mass, ' l1 ': logL, ' te1 ': logT}
        with open(filename, 'r') as inp:
            names = replace_(inp.readline().strip(), rdict).split()
            self.data = np.genfromtxt(filename, names=names)


def check(fname):
    t = AGBTrack(fname)
    fig, ax = plt.subplots()
    x = t.data[logT]
    y = t.data[logL]
    ax.plot(x, y)
    itp = np.nonzero(np.isfinite(t.data['status']))[0][0]
    
    ax.plot(x[itp], y[itp], 'o')
    xoff = 0.005
    yoff = 0.1
    #ax.set_xlim(x[itp] - xoff, x[itp] + xoff)
    #ax.set_ylim(y[itp] - yoff, y[itp] + yoff)
    ax.set_title('{:.4f} {:.3f}'.format(t.Z, t.mass))
    plt.draw()

def combine_parsec_colibri(diag=False, agb_track_loc=None, prc_track_loc=None,
                           first_tp_loc=None, ptcri_loc=None):
    def getpmasses(strings):
        snames = [os.path.split(s)[1] for s in strings]
        return np.array(['.'.join(s.split('_M')[1].split('.')[:2])
                         for s in snames], dtype=float)
        
    def getcmasses(strings):
        
        snames = [os.path.split(s)[1] for s in strings]
        return np.array([s.split('_')[1] for s in snames], dtype=float)

    agb_track_loc = '/Users/rosenfield/Desktop/S_NOV13'
    prc_track_loc = '/Users/rosenfield/Dropbox/CAF09_V1.2S_M36_LT/tracks'
    first_tp_loc = '/Users/rosenfield/Dropbox/CAF09_V1.2S_M36_LT/tpagb'
    ptcri_loc = '/Users/rosenfield/Dropbox/CAF09_V1.2S_M36_LT/data'
    
    outputloc = prc_track_loc
    firsttps = get_files(first_tp_loc, '*.INP')
    for firsttp in firsttps:
        onetp = FirstTP(firsttp)
        masses = onetp.data[mass]
        if onetp.Z == 0.05:
            continue
        # get the ptcri files
        try:
            ptcri_files = get_files(ptcri_loc, 'p2m*{:g}Y*'.format(onetp.Z))
        except ValueError:
            print(sys.exc_info()[1])
            continue
        
        ptcri_filepms, = [p for p in ptcri_files if not 'hb' in p.lower()]
        ptcri_filehb, = [p for p in ptcri_files if 'hb' in p.lower()]
        
        # get the agb tracks
        agb_track_dir, = get_dirs(agb_track_loc, '{:g}_'.format(onetp.Z))
        all_agb_tracks = np.array(get_files(agb_track_dir, 'agb_*_Mdot50*dat'))
        agb_masses = getcmasses(all_agb_tracks)
        max_agbmass = max(agb_masses)
        
        # get the parsec tracks, but only the ones that could have TP-AGB
        prc_track_dir, = get_dirs(prc_track_loc, '{:g}Y'.format(onetp.Z))
        all_prc_tracks = np.array(get_files(prc_track_dir, '*.DAT'))
        hb_tracks = np.array([h for h in all_prc_tracks if 'hb' in h.lower()])
        nonhb_tracks = np.array([h for h in all_prc_tracks if not 'hb' in h.lower()])
        
        max_hbmass = np.max(getpmasses(hb_tracks))
        nonhb_masses = getpmasses(nonhb_tracks)
        # Get the HB (not the low mass up to RGB of the same mass)
        iprch = np.array([i for i, p in enumerate(nonhb_masses) if p > max_hbmass])
        
        # Cut out masses greather than are available to the TP-AGB
        iprca = np.array([i for i, p in enumerate(nonhb_masses[iprch]) if p <= max_agbmass])
        prc_tracks = np.concatenate([hb_tracks, nonhb_tracks[iprch[iprca]]])
        prc_masses = getpmasses(prc_tracks)
        
        common_masses = np.sort(list(set(masses) & set(prc_masses) & set(agb_masses)))
        
        
        new_track_dir = os.path.join(outputloc, os.path.split(prc_track_dir)[1])
        ensure_dir(new_track_dir)
        #fmt = 'Z{:.4f}_Y{:.3f}_M{:.3f}.dat'
        print(new_track_dir)
        for mass_ in common_masses:
            iprc, = np.where(prc_masses == mass_)[0]
            parsec_track = prc_tracks[iprc]
            ptcri_file = ptcri_filepms
            if 'hb' in parsec_track.lower():
                ptcri_file = ptcri_filehb
            
            try:
                parsec = Track(parsec_track, ptcri_file=ptcri_file,
                               ptcri_kw={'sandro': False})
            except:
                print(sys.exc_info()[1])
                continue                
            iagb, = np.where(agb_masses == mass_)[0]
            colibri_track = all_agb_tracks[iagb]
            colibri = AGBTrack(colibri_track)

            iotp, = np.where(onetp.data[mass] == mass_)[0]
            onetpm = onetp.data[iotp]

            #fname = fmt.format(parsec.Z, parsec.Y, mass_)
            output = os.path.join(new_track_dir, parsec_track + '.TPAGB')
            attach(parsec, colibri, onetpm, output, diag=diag)


def attach(parsec, colibri, onetpm, output, diag=True):
    assert parsec.Z == colibri.Z == onetpm['z0'], 'Metallicity mismatch'
    assert parsec.mass == colibri.mass, 'Mass mismatch'
    
    ifin = parsec.iptcri[-1]
    ipmatch = np.argmin(abs(parsec.data[:ifin][logL] - onetpm[logL]))
    
    # Add the track age to tp-agb age (which starts at 0.0)
    colibri.data[age] += parsec.data[ipmatch][age]
    
    # icmatch = 0 by design, but just in case:
    icmatch = np.argmin(abs(colibri.data[logL] - onetpm[logL]))
    
    ptrack = pd.DataFrame(parsec.data[:ipmatch - 1])
    ctrack = pd.DataFrame(colibri.data[icmatch:])
    
    all_data = pd.DataFrame()
    all_data = all_data.append(ptrack, ignore_index=True)
    all_data = all_data.append(ctrack, ignore_index=True)
    
    all_data.to_csv(output, sep=' ', na_rep='nan', index=False)
    print('wrote to {}'.format(output))
    if diag:
        print_diffs(colibri.data[icmatch],
                    parsec.data[:ifin][ipmatch])
    
    return 


def print_diffs(cval, pval):
    fmt = '{:.4f} {:.4f} {:.5f} {:.5f}'
    #print('PARSEC idx: {} COLIBRI idx: {}'.format(ipmatch, icmatch))
    dlogL = cval[logL] - pval[logL]
    dlogT = cval[logT] - pval[logT]
    print(fmt.format(cval['Z'], pval[mass], dlogL, dlogT))


