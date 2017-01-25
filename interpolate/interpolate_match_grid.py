import argparse
import os
import pdb
import sys
import seaborn

try:
    import trilegal
except ImportError:
    print('May need IsoTrack reader')
    pass

import matplotlib.pylab as plt
import numpy as np

from .. import fileio
from ..eep.critical_point import Eep

seaborn.set()

__all__ = ['interp_match_grid', 'interp_mhefs']


def reformat_filename(fname, fmt='Z{:.4f}_Y{:.3f}_M{:.3f}{}'):
    """
    Common filename formats
    e.g.,
    os.system('mv {} {}'.format(fname, reformat_filename(fname))
    """
    z = float(fname.split('Z')[1].split('Y')[0].replace('_', ''))
    y = float(fname.split('Y')[1].split('O')[0].replace('_', ''))
    mstr = fname.split('M')[1].split('.dat')[0].replace('.HB', '')
    ext = fname.split('M'+mstr)[1]
    m = float(mstr)
    return fmt.format(z, y, m, ext)


def plot_MheF(isotracks=None, labels=None, colors=None):
    """ plot the minimum initial mass for He Fusion """
    if isotracks is None:
        print('WARNING: attempting to use hard coded isotracks')
        isotracks = ['isotrack/parsec/CAF09_MC_S13v3_OV0.3.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.4.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.5.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.6.dat',
                     'isotrack/parsec/CAF09_S12D_NS_1TP.dat']
        isotracks = [os.path.join(os.environ['TRILEGAL_ROOT'], i)
                     for i in isotracks]

    if labels is None:
        labels = ['$\Lambda_c=0.3$',
                  '$\Lambda_c=0.4$',
                  '$\Lambda_c=0.5$',
                  '$\Lambda_c=0.6$',
                  '$S12D\_NS\_1TP$']
    if colors is None:
        colors = ['darkred', 'orange', 'navy', 'purple', 'k']

    fig, ax = plt.subplots()
    for i, isotrack in enumerate(isotracks):
        isot = trilegal.IsoTrack(isotrack)
        ax.plot(isot.Z, isot.mhefs, lw=2, label=labels[i], color=colors[i])
        ax.plot(isot.Z, isot.mhefs, 'o', color=colors[i])

    ax.grid()
    ax.set_xlim(0.001, 0.0085)
    ax.set_ylim(1.55, 2.05)
    return ax


def interp_mhefs(isodirs, outfile=None):
    """
    Write the minimum initial mass for He fusion to a file, interpolating
    between isotracks.

    Parameters
    ----------
    isotracks : list of strings
        path to parsec isotrack files.
        eg:
        isodirs = ['CAF09_MC_S13v3_OV0.3',
                   'CAF09_MC_S13v3_OV0.4',
                   'CAF09_MC_S13v3_OV0.5',
                   'CAF09_MC_S13v3_OV0.6',
                   'CAF09_MC_S13v3_OV0.7']
    outfile : string
        filename of output file
    """
    def pretty(ov, marr):
        """ make a %.2f string combining a float and an array """
        return ' '.join(['%.2f' % i
                         for i in np.concatenate(([ov], marr))]) + '\n'

    outfile = outfile or 'MHeF_interp.dat'

    line = ''
    mhefs = np.array([])
    ovs = np.array([])
    zs = np.array([])
    for isodir in isodirs:
        # each INT file in each dir
        int_files = fileio.get_files(isodir, '*INT*')
        if len(int_files) == 0:
            print('no INT files found in {0:s}'.format(isodir))
            continue
        for int_file in int_files:
            z = float(os.path.split(int_file)[1].split('_Z')[1].split('_')[0])
            zs = np.append(zs, z)
            with open(int_file, 'r') as inp:
                lines = inp.readlines()
            # MHeF is the first mass of the INT* files issue is where the
            # mass is in the file
            if int_file.endswith('2'):
                # Leo's formatted isotrack files *INT2
                nsplits = int(lines[0])
                iline = nsplits + 3
            else:
                # Sandro's dbert/*INT files
                iline = -1

            mhefs = np.append(mhefs, float(lines[iline].split()[0]))

        ov = float(os.path.split(int_file)[1].split('_OV')[1].split('_')[0])
        ovs = np.append(ovs, ov)

    zs = np.unique(zs)
    # one mass for one Z one row for each OV.
    mhefs = mhefs.reshape(len(mhefs) // len(zs), len(zs))

    line += '# OV ' + ' '.join(['Z%g' % z for z in zs]) + '\n'

    # midpoint interpolation
    for i in range(len(isodirs)-1):
        intov = (ovs[i+1] + ovs[i]) / 2.
        intpd = (mhefs[i+1] + mhefs[i]) / 2.
        line += pretty(ovs[i], mhefs[i])
        line += pretty(intov, intpd)
    line += pretty(ovs[i+1], mhefs[i+1])

    with open(outfile, 'w') as outf:
        outf.write(line)
    print(('wrote %s' % outfile))
    return outfile


def interpolate_between_sets(match_dir1, match_dir2, outdir, mhef,
                             overwrite=False, plot=False,
                             truth_track_loc='', frac=2.):
    def strip_m(s):
        return float(s.split('_M')[-1].replace('.dat', '').replace('.HB', ''))

    def strip_z(s):
        return float(s.split('Z')[-1].split('Y')[0].replace('_', ''))

    def get_names(s):
        return [os.path.split(i)[1] for i in s]

    header = 'logAge Mass logTe Mbol logg C/O'
    fileio.ensure_dir(outdir)
    t1files = sorted(fileio.get_files(match_dir1, '*.dat'),
                     key=lambda t: strip_m(t))
    t2files = sorted(fileio.get_files(match_dir2, '*.dat'),
                     key=lambda t: strip_m(t))

    tname1s = get_names(t1files)
    tname2s = get_names(t2files)

    t1hbs = [t for t in t1files if 'HB' in t]
    t2hbs = [t for t in t2files if 'HB' in t]
    i2s = [i for i, t in enumerate(tname2s) if t in tname1s]
    t2files = np.array(t2files)[i2s]
    i1s = [i for i, t in enumerate(tname1s) if t in tname2s]
    t1files = np.array(t1files)[i1s]

    tname1s = get_names(t1files)
    tname2s = get_names(t2files)
    ntracks = len(t1files)

    # assert tname1s == tname2s, 'Track mismatches'
    if tname1s != tname2s:
        print('Track mismatches')
        pdb.set_trace()

    t1s = [np.loadtxt(t) for t in t1files]
    t2s = [np.loadtxt(t) for t in t2files]
    for i in range(ntracks):
        if plot:
            if os.path.isdir(truth_track_loc):
                gs1, gs2, [lax, lbax, lrax], [rax, rbax, rrax] = \
                    setup_diagplot()
            else:
                fig, (lax, rax) = plt.subplots(nrows=2)

        addedhb = False  # for plotting
        mass = strip_m(t1files[i])
        # simple mid point interpolation
        nt1s = len(t1s[i])
        nt2s = len(t2s[i])

        # most of the time both tracks are the same length
        if nt1s == nt2s:
            track = (t1s[i] + t2s[i]) / frac
        else:
            i1, i2 = np.argsort([nt1s, nt2s])
            nt1, nt2 = [nt1s, nt2s][i1], [nt1s, nt2s][i2]
            t1, t2 = [t1s[i], t2s[i]][i1], [t1s[i], t2s[i]][i2]
            tname1, tname2 = [tname1s[i], tname2s[i]][i1], [tname1s[i], tname2s[i]][i2]
            thbs = t2hbs
            if tname1 == tname1s[i]:
                thbs = t1hbs
            if mass <= mhef:
                # keep the track short.
                track = (t1 + t2[:nt1]) / frac
                print('tuncating HB', mass, nt1, nt2, i)
            else:
                # add HB to track 1
                print('adding HB', mass, len(t1s[i]), len(t2s[i]))
                thb, = [t for t in thbs if 'M%.2f' % mass in t]
                thb = np.genfromtxt(thb)
                twithhb = rg_tip_heb_transition(thb, t1)
                track = (twithhb + t2) / frac
                addedhb = True

        if plot:
            _plot(track, mass, lax, rax, label='interp')
            if addedhb:
                _plot(twithhb, '', lax, rax, label='added hb')
                _plot(thb, '', lax, rax, label='hb')

            _plot(t1s[i], '', lax, rax, label='t1')
            _plot(t2s[i], '', lax, rax, label='t2')

            if os.path.isdir(truth_track_loc):
                z = strip_z(tname1s[i])
                tds = fileio.get_dirs(truth_track_loc)
                td, = [t for t in tds if str(z) in t]
                mstr = '{0:.2f}'.format(mass)
                if mass < 1.:
                    mstr = mstr[1:]
                truth_tracks = \
                    fileio.get_files(td, '*Z{0:.4f}*M{1:s}*'.format(z, mstr))
                if len(truth_tracks) > 0:
                    if len(truth_tracks) > 1:
                        truet0 = np.loadtxt(truth_tracks[0])
                        truet1 = np.loadtxt(truth_tracks[1])
                        if len(truet0) == len(track):
                            truet = truet0
                        elif len(truet1) == len(track):
                            truet = truet1
                        elif len(truet0[:nt1s]) == len(track):
                            truet = truet0[:nt1s]
                        else:
                            pdb.set_trace()
                    else:
                        truet = np.loadtxt(truth_tracks[0])
                        if len(truet) != len(track):
                            if len(truet[:nt1s]) == len(track):
                                truet = truet[:nt1s]
                    try:
                        _plot(truet, '', lax, rax,  label='truth')
                        diff_plot(truet, track, lbax, rbax, lrax, rrax)
                        rbax.set_xscale('log')
                    except:
                        print('hey!')
                        import pdb; pdb.set_trace()

            rax.legend(loc='best')
            figname = os.path.join(outdir, tname1s[i].replace('dat', 'png'))

            plt.savefig(figname)
            # print('wrote {}'.format(figname))
            plt.close()
        outfile = os.path.join(outdir, tname1s[i])
        if not os.path.isfile(outfile) or overwrite:
            # np.savetxt(outfile, track, header=header, fmt='%.8f')
            np.savetxt(outfile, track, header=header, fmt='%.8f')
        # print('wrote {}'.format(outfile))


def rg_tip_heb_transition(hb_track, track):
    """
    Attach a HB model to a PMS model.
    Done in a consistent way as in TRILEGAL. Basically, zero time goes by,
    linear connection. The idea is that no stars should fall in this region
    because there are simply no tracks calculated. If you want a track
    following a hiashi line, well, calculate one. If you're interpolating,
    you're gonna have a slight error on a time scale of 1e5 years, counting a
    star that could have been in a transition phase from RG_TIP to HE_BEG as a
    RGB star. At this point in time, a negligable error.
    """
    eep = Eep()
    ntrans = eep.trans
    rg_tip = eep.nms - 1

    agei = 100.
    agef = 10 ** hb_track.T[0][0]

    te0, tef = track.T[2][rg_tip], hb_track.T[2][0]
    mbol0, mbolf = track.T[3][rg_tip], hb_track.T[3][0]
    m, b = np.polyfit([te0, tef], [mbol0, mbolf], 1)

    age = np.linspace(agei, agef, ntrans, endpoint=False)
    logte = np.linspace(te0, tef, ntrans, endpoint=False)
    Mbol = m * logte + b
    mass = np.zeros(ntrans) + track.T[1][0]
    logg = -10.616 + np.log10(mass) + 4.0 * logte - (4.77 - Mbol) / 2.5
    CO = np.zeros(ntrans)
    logage = np.log10(10 ** track.T[0][rg_tip] + age)
    trans_track = np.column_stack([logage, mass, logte, Mbol, logg, CO])

    hb_track.T[0] = np.log10(10 ** hb_track.T[0] + 10 ** logage[-1])
    new_track = np.concatenate((track, trans_track, hb_track))
    return new_track


def setup_diagplot():
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12, 8))
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(left=0.1, right=0.48, wspace=0.05)
    lax = plt.subplot(gs1[:-1, :2])
    lbax = plt.subplot(gs1[-1, :2])
    lrax = plt.subplot(gs1[:-1, -1])
    lbax.set_xlabel(r'$\log\ Te$')
    lax.set_ylabel(r'$\rm{Mbol}$')
    gs2 = gridspec.GridSpec(3, 3)
    gs2.update(left=0.6, right=0.98, hspace=0.05)
    rax = plt.subplot(gs2[:-1, :2])
    rbax = plt.subplot(gs2[-1, :2])
    rrax = plt.subplot(gs2[:-1, -1])
    rbax.set_xlabel(r'$\rm{Age}$')
    return gs1, gs2, [lax, lbax, lrax], [rax, rbax, rrax]


def diff_plot(track1, track2, lbax, rbax, lrax, rrax):
    from matplotlib.ticker import MaxNLocator
    lxdiff = track1.T[2] - track2.T[2]
    rxdiff = 10 ** track1.T[0] - 10 ** track2.T[0]
    ydiff = track1.T[3] - track2.T[3]
    lbax.plot(track1.T[2], lxdiff, '.')
    rbax.plot(10 ** track1.T[0], lxdiff, '.')
    [ax.axhline(0.) for ax in [lbax, rbax]]
    for ax in lrax, rrax:
        ax.axvline(0.)
        ax.plot(ydiff, track1.T[3], '.')
        ax.xaxis.set_major_locator(MaxNLocator(5))


def _plot(track, mass, lax, rax, label=''):
    alpha = 0.3
    if len(label) > 0:
        alpha = 1.
    lax.plot(track.T[2], track.T[3], alpha=alpha)
    if mass != '':
        rax.plot(10 ** track.T[0], track.T[3], alpha=alpha,
                 label='${}\ {:.2f}$'.format(label, mass))
    else:
        rax.plot(10 ** track.T[0], track.T[3], label=label,
                 alpha=alpha)

    return


def read_mhef(mhef_file):
    with open(mhef_file, 'r') as inp:
        lines = inp.readlines()
    zs = np.array([l.replace('Z', '')
                   for l in lines[0].split() if 'Z' in l], dtype=float)
    data = np.genfromtxt(mhef_file)
    return data, zs


def interp_match_grid(dir1, dir2, mhef_file, overwrite=False,
                      plot=False, truth_track_loc='', newsubs=None):
    frac = 2
    data, zs = read_mhef(mhef_file)
    subs = [dir1, dir2]
    pts = np.sort(np.array([s.replace('ov', '').replace('/', '') for s in subs],
                           dtype=float))
    if newsubs is None:
        interps = data.T[0][(data.T[0] > pts[0]) & (data.T[0] < pts[1])]
        if len(interps) == 0:
            if frac == 2:
                interps = np.array([np.mean(pts)])
        newsubs = ['ov{:.2f}'.format(s) for s in interps]
    else:
        newsubs = np.asarray([newsubs])
        interps = np.array([s.replace('ov', '').replace('/', '')
                            for s in newsubs], dtype=float)
    print('interpolate for these new values: {}'.format(interps))
    sets = [[os.path.join(s, l) for l in os.listdir(s)
             if not l.startswith('.') and os.path.isdir(os.path.join(s, l))]
            for s in subs]
    # frac=2 default: mean would assume we're finding the point inbetween.
    for i in range(len(sets)-1):
        for j in range(len(sets[i])):
            newset = os.path.split(sets[i][j])[1]
            newset = newset.replace('OV{:.1f}'.format(pts[i]),
                                    'OV{:.2f}'.format(interps[i]))
            newdir = os.path.join(newsubs[i], newset)
            print('interpolating output: {0:s}'.format(newdir))
            interpolate_between_sets(sets[i][j], sets[i+1][j], newdir,
                                     data[2*i+1][j+1], plot=plot, frac=frac,
                                     truth_track_loc=truth_track_loc)
    return


def main(argv):
    """
    Report ... quick test between OV0.4  OV0.6 to compare to parsec:
    Even the low mass where nothing should change was off. NEED TO CALC OV0.5

    quick test between ov0.30 and 0.60:
        Some offsets likely due to the end of the track differences.
        Other offsets because comparing to ov0.40 isn't correct, this will
        create ov0.45.
        Lots of structure on HB phase looks pretty strange. It might be better
        NOT to interpolate and run with MATCH but use a KDE later.

    """
    parser = argparse.ArgumentParser(description=" ")

    parser.add_argument('-m', '--mhef_file', type=str,
                        help='file containing the He fusion masses')

    parser.add_argument('-n', '--newsubs', type=str,
                        help='new subdirectory name')

    parser.add_argument('-i', '--isodir_loc', type=str,
                        help='where the isotrack files are (if not -m)')

    parser.add_argument('-t', '--truth_track_loc', type=str, default='',
                        help='over plot comparison tracks from this location')

    parser.add_argument('-d', '--diag_plot', action='store_true',
                        help='make HB plots')

    parser.add_argument('-v', '--pdb', action='store_true',
                        help='invoke python debugger')

    parser.add_argument('dir1', type=str, help='directory 1')
    parser.add_argument('dir2', type=str, help='directory 2')

    args = parser.parse_args(argv)

    if args.pdb:
        pdb.set_trace()

    # Where are the INT files
    if not args.mhef_file:
        isodir_loc = args.isodir_loc or os.getcwd()
        isodirs = [os.path.join(isodir_loc, l) for l in os.listdir(isodir_loc) if os.path.isdir(os.path.join(isodir_loc, l))]
        args.mhef_file = interp_mhefs(isodirs)

    interp_match_grid(args.dir1, args.dir2,
                      args.mhef_file,
                      plot=args.diag_plot,
                      truth_track_loc=args.truth_track_loc,
                      newsubs=args.newsubs)

if __name__ == '__main__':
    main(sys.argv[1:])
