"""Modules to attach COLIBRI to PARSEC"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

from .config import *
from .fileio import get_files, get_dirs, ensure_dir
from .utils import replace_, get_zy
from .tracks.track import Track, AGBTrack

seaborn.set()


class FirstTP(object):
    """Input files between PARSEC and COLIBRI"""
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

    itp = np.nonzero(np.isfinite(t.data['status']))[0][0]

    ax.plot(x, y)
    ax.plot(x[:itp], y[:itp])
    ax.plot(x[itp:], y[itp:])
    ax.plot(x[itp], y[itp], 'o')

    xoff = 0.005
    yoff = 0.1
    ax.set_xlim(x[itp] - xoff, x[itp] + xoff)
    ax.set_ylim(y[itp] - yoff, y[itp] + yoff)
    ax.set_title('{:.4f} {:.3f}'.format(t.Z, t.mass))
    plt.draw()


def getpmasses(strings):
    """get PARSEC mass"""
    snames = [os.path.split(s)[1] for s in strings]
    return np.array(['.'.join(s.split('_M')[1].split('.')[:2])
                     for s in snames], dtype=float)


def getcmasses(strings):
    """get COLIBRI mass"""
    snames = [os.path.split(s)[1] for s in strings]
    return np.array([s.split('_')[1] for s in snames], dtype=float)


def _getptcri(ptcri_loc, z):
    """get ptcri file names (HB and non HB) at a given Z"""
    try:
        ptcri_files = get_files(ptcri_loc, 'p2m*{:g}Y*'.format(z))
    except ValueError:
        print(sys.exc_info()[1])

    ptcri_filepms, = [p for p in ptcri_files if 'hb' not in p.lower()]
    ptcri_filehb, = [p for p in ptcri_files if 'hb' in p.lower()]
    return ptcri_filepms, ptcri_filehb


def _getagbtracks(agb_track_loc, z):
    """ get all agb track names at a given Z"""
    agb_track_dir, = get_dirs(agb_track_loc, '{:g}_'.format(z))
    all_agb_tracks = np.array(get_files(agb_track_dir, 'agb_*_Mdot50*dat'))
    return all_agb_tracks


def _getparsectracks(prc_track_loc, z, maxmass):
    """get all parsec track names at a given Z up to max mass"""
    prc_track_dir, = get_dirs(prc_track_loc, '{:g}Y'.format(z))
    all_prc_tracks = np.array(get_files(prc_track_dir, '*.DAT'))
    hb_tracks = np.array([h for h in all_prc_tracks if 'hb' in h.lower()])
    nonhb_tracks = np.array([h for h in all_prc_tracks
                             if 'hb' not in h.lower()])

    max_hbmass = np.max(getpmasses(hb_tracks))
    nonhb_masses = getpmasses(nonhb_tracks)
    # Cull the non-HB (not the lowest mass tracks with not TPAGB)
    iprch = np.array([i for i, p in enumerate(nonhb_masses)
                      if p > max_hbmass])

    # Cut out masses greather than are available to the TP-AGB
    iprca = np.array([i for i, p in enumerate(nonhb_masses[iprch])
                      if p <= maxmass])
    try:
        prc_tracks = np.concatenate([hb_tracks, nonhb_tracks[iprch[iprca]]])
    except:
        prc_tracks = []
    return prc_tracks


def combine_parsec_colibri(diag=False, agb_track_loc=None, prc_track_loc=None,
                           first_tp_loc=None, ptcri_loc=None, outputloc=None,
                           overwrite=False):
    """Combine PARSEC and COLIRBI (call attach)"""
    line = ''
    firsttps = get_files(first_tp_loc, '*.INP')
    for firsttp in firsttps:
        onetp = FirstTP(firsttp)
        if onetp.Z == 0.05:
            # (Currently no PARSEC models at Z=0.05)
            continue

        # get the ptcri files
        ptcri_filepms, ptcri_filehb = _getptcri(ptcri_loc, onetp.Z)

        # get the agb tracks
        all_agb_tracks = _getagbtracks(agb_track_loc, onetp.Z)

        # Colibri masses
        agb_masses = getcmasses(all_agb_tracks)
        max_agbmass = max(agb_masses)

        # get the parsec tracks
        prc_tracks = _getparsectracks(prc_track_loc, onetp.Z, max_agbmass)
        if len(prc_tracks) == 0:
            continue

        # 1TP masses
        otp_masses = onetp.data[mass]
        # PARSEC masses
        prc_masses = getpmasses(prc_tracks)

        common_masses = \
            np.sort(list(set(otp_masses) & set(prc_masses) & set(agb_masses)))

        new_track_dir = \
            os.path.join(outputloc,
                         os.path.split(os.path.split(prc_tracks[0])[0])[1])
        ensure_dir(new_track_dir)
        # fmt = 'Z{:.4f}_Y{:.3f}_M{:.3f}.dat'
        print(new_track_dir)
        for mass_ in common_masses:
            iprc, = np.where(prc_masses == mass_)[0]
            iagb, = np.where(agb_masses == mass_)[0]
            iotp, = np.where(otp_masses == mass_)[0]

            parsec_track = prc_tracks[iprc]
            colibri_track = all_agb_tracks[iagb]
            onetpm = onetp.data[iotp]
            ptcri_file = ptcri_filepms

            output = os.path.join(new_track_dir, parsec_track + '.TPAGB')
            if os.path.isfile(output) and not overwrite:
                print('not overwriting {}'.format(output))
                continue
            if 'hb' in parsec_track.lower():
                ptcri_file = ptcri_filehb

            # load PARSEC Track
            try:
                parsec = Track(parsec_track, ptcri_file=ptcri_file,
                               ptcri_kw={'sandro': False})
            except ValueError as e:
                print('Problem with {}'.format(parsec_track))
                print(e)
                continue
            # Load COLIBRI Track
            colibri = AGBTrack(colibri_track)
            # First two lines are 0 age
            colibri.data[age].iloc[1] = colibri.data[age].iloc[2]/2
            # fname = fmt.format(parsec.Z, parsec.Y, mass_)
            line += attach(parsec, colibri, onetpm, output, diag=diag)
        if diag:
            with open('colibri_attach.log', 'w') as outp:
                outp.write(line)


def radius(x1, y1, x2, y2):
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return np.argmin(dist), np.min(dist)


def attach(parsec, colibri, onetpm, output, diag=True):
    """
    Adopt PARSEC until a point in the TP-AGB with the same luminosity as 1TP
    and a Teff differing by less than 100 K.
    """
    line = ''
    assert parsec.Z == colibri.Z == onetpm['z0'], 'Metallicity mismatch'
    assert parsec.mass == colibri.mass, 'Mass mismatch'
    try:
        ifin = np.max(parsec.iptcri)  # final track eep
        ifin = -1
    except AttributeError as e:
        return

    ipmatch, _ = radius(parsec.data[:ifin][logT], parsec.data[:ifin][logL],
                        onetpm[logT], onetpm[logL])
    # Add the track age to tp-agb age (which starts at 0.0)
    # print(''.join(['{} {} {} {}\n'.format(col,
    #                                       colibri.data[col].iloc[0],
    #                                       colibri.data[col].iloc[1],
    #                                       colibri.data[col].iloc[2])
    #                for col in colibri.data.columns]))
    colibri.data[age] += parsec.data[ipmatch][age]

    inds, = np.nonzero(colibri.data['NTP'] < 2)

    im, _ = radius(colibri.data[logT].iloc[inds],
                   colibri.data[logL].iloc[inds],
                   onetpm[logT], onetpm[logL])
    icmatch = inds[im]

    if icmatch != 0:
        ntp = colibri.data['NTP'][icmatch]
        print('Warning: might be missing some TP-AGB: ',
              'Colibri matched index={}, NTP={}'.format(icmatch, ntp))

    lmatch = parsec.data[logL][ipmatch] - colibri.data[logL].iloc[icmatch]
    tmatch = 10 ** parsec.data[logT][ipmatch] - \
        10 ** colibri.data[logT].iloc[icmatch]
    if np.abs(tmatch) > 100 or np.abs(lmatch) > 0.1:
        err = 'bad 1TP? {} {} {} {}'.format(parsec.Z, parsec.mass, lmatch,
                                            tmatch)
        print(err)
        return err
    ptrack = pd.DataFrame(parsec.data[:ipmatch - 1])
    ctrack = pd.DataFrame(colibri.data.iloc[icmatch:])

    all_data = pd.DataFrame()
    all_data = all_data.append(ptrack, ignore_index=True)
    all_data = all_data.append(ctrack, ignore_index=True)

    all_data.to_csv(output, sep=' ', na_rep='nan', index=False)
    # print('wrote to {}'.format(output))
    if diag:
        # print('PARSEC idx: {} COLIBRI idx: {}'.format(ipmatch, icmatch))
        line += print_diffs(colibri.data.iloc[icmatch],
                            parsec.data[:ifin][ipmatch])
        plot_hrd(all_data, colibri, parsec, icmatch, ipmatch, onetpm, output,
                 lmatch=lmatch, tmatch=tmatch)
    return line


def plot_hrd(all_data, colibri, parsec, icmatch, ipmatch, onetpm, output,
             outdir=None, lmatch=None, tmatch=None):
    if outdir is None:
        outdir = 'colibi_parsec'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    output = os.path.join(outdir, *os.path.split(output)[1:])

    fig, ax = plt.subplots()
    xcol, ycol = logT, logL
    lab = ['colibri', 'parsec']
    ax.plot(onetpm[xcol], onetpm[ycol], 'o', ms=10, alpha=0.4, label='1TP')
    ax.plot(all_data[xcol], all_data[ycol], lw=3, alpha=0.4)
    for i, (track, idx) in enumerate(zip([colibri, parsec],
                                         [icmatch, ipmatch])):
        xdata = track.data[xcol]
        ydata = track.data[ycol]
        l, = ax.plot(xdata, ydata, label=lab[i])
        try:
            ax.plot(xdata[idx], ydata[idx], 'o', color=l.get_color())
        except:
            ax.plot(xdata.iloc[idx], ydata.iloc[idx], 'o', color=l.get_color())

    # xoff = 0.05
    # yoff = 0.1
    # ax.set_xlim(xdata[idx] - xoff, xdata[idx] + xoff)
    # ax.set_ylim(ydata[idx] - yoff, ydata[idx] + yoff)
    ax.set_xlim(np.max(colibri.data[xcol]), np.min(colibri.data[xcol]))
    ax.set_ylim(np.min(colibri.data[ycol]), np.max(colibri.data[ycol]))
    title = '{:.4f} {:.3f}'.format(track.Z, track.mass)
    if lmatch is not None:
        title += ' logL(P-C)={:.3f}'.format(lmatch)
    if tmatch is not None:
        title += ' T(P-C)={:.3f}'.format(tmatch)
    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    plt.legend(loc='best')
    plt.savefig(output + '.pdf')
    print('wrote {}.pdf'.format(output))
    plt.close()


def print_diffs(cval, pval):
    fmt = '{:.4f} {:.4f} {:.5f} {:.5f}\n'
    dlogL = cval[logL] - pval[logL]
    dlogT = cval[logT] - pval[logT]
    return fmt.format(cval['Z'], pval[mass], dlogL, dlogT)


def main(argv):

    parser = argparse.ArgumentParser(description="Attach COLIBRI to PARSEC")

    parser.add_argument('agb_track_loc', type=str, default=os.getcwd(),
                        help='path to AGB tracks')

    parser.add_argument('prc_track_loc', type=str, default=os.getcwd(),
                        help='path to PARSEC tracks')

    parser.add_argument('first_tp_loc', type=str, default=os.getcwd(),
                        help='path to 1TP files')

    parser.add_argument('ptcri_loc', type=str, default=os.getcwd(),
                        help='path to ptcri files')

    parser.add_argument('--outputloc', type=str,
                        help='output location (default first_tp_loc)')

    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing output files')

    parser.add_argument('-d', '--diag', action='store_true',
                        help='toggle diagnostics')

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='toggle python debugger')

    args = parser.parse_args(argv)

    if args.outputloc is None:
        args.outputloc = args.first_tp_loc

    if args.pdb:
        import pdb
        pdb.set_trace()
    combine_parsec_colibri(agb_track_loc=args.agb_track_loc,
                           prc_track_loc=args.prc_track_loc,
                           first_tp_loc=args.first_tp_loc,
                           ptcri_loc=args.ptcri_loc,
                           outputloc=args.outputloc,
                           diag=args.diag,
                           overwrite=args.overwrite)

if __name__ == "__main__":
    main(sys.argv[1:])
