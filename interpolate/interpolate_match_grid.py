import argparse
import matplotlib.pylab as plt
import numpy as np
import os
import trilegal
from ResolvedStellarPops import fileio
import sys
from ..eep.critical_point import Eep

__all__ = ['interp_match_grid', 'interp_mhefs']

def plot_MheF(isotracks=None, labels=None, colors=None):
    """ plot the minimum initial mass for He Fusion """
    if isotracks is None:
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
        path to parsec isotrack files. Must be in order!
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
        return ' '.join(['%.2f' % i for i in np.concatenate(([ov], marr))]) + '\n'

    outfile = outfile or 'MHeF_interp.dat'

    line = ''
    mhefs = np.array([])
    ovs = np.array([])
    zs = np.array([])
    for isodir in isodirs:
        # each INT file in each dir
        int_files = fileio.get_files(isodir, '*INT*')
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
        ovs  = np.append(ovs, ov)

    zs = np.unique(zs)
    # one mass for one Z one row for each OV.
    mhefs = mhefs.reshape(len(mhefs)/len(zs), len(zs))

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
    print('wrote %s' % outfile)
    return outfile



def interpolate_between_sets(match_dir1, match_dir2, outdir, mhef,
                             overwrite=False, plot=False):
    def strip_m(s):
        return float(s.split('7_M')[-1].replace('.dat', '').replace('.HB',''))

    def get_names(s):
        return [os.path.split(i)[1] for i in s]

    header = '# logAge Mass logTe Mbol logg C/O \n'
    fileio.ensure_dir(outdir)
    t1files = sorted(fileio.get_files(match_dir1, '*.dat'),
                     key=lambda t: strip_m(t))
    t2files = sorted(fileio.get_files(match_dir2, '*.dat'),
                     key=lambda t: strip_m(t))

    tname1s = get_names(t1files)
    tname2s = get_names(t2files)

    t1hbs = [t for t in t1files if 'HB' in t]

    i2s = [i for i, t in enumerate(tname2s) if t in tname1s]
    t2files = np.array(t2files)[i2s]
    i1s = [i for i, t in enumerate(tname1s) if t in tname2s]
    t1files = np.array(t1files)[i1s]

    tname1s = get_names(t1files)
    tname2s = get_names(t2files)
    ntracks = len(t1files)

    #assert tname1s == tname2s, 'Track mismatches'
    if tname1s != tname2s:
        print('Track mismatches')
        import pdb; pdb.set_trace()

    t1s = [np.loadtxt(t) for t in t1files]
    t2s = [np.loadtxt(t) for t in t2files]
    for i in range(ntracks):
        addedhb = False  # for plotting
        mass = strip_m(t1files[i])
        # simple mid point interpolation
        try:
            track = (t1s[i] + t2s[i]) / 2.
        except:
            nt1s = len(t1s[i])
            if mass < mhef:
                # shorter track
                track = (t1s[i] + t2s[i][:nt1s]) / 2.
                print 'shorter', mass, len(t1s[i]), len(t2s[i]), i
            if mass >= mhef:
                # longer track
                t1hb, = [t for t in t1hbs if 'M%.2f' % mass in t]
                t1hb = np.genfromtxt(t1hb)
                print 'longer', mass, len(t1s[i]), len(t2s[i])
                # add the transition points
                t1withhb = rg_tip_heb_transition(t1hb, t1s[i])
                track = (t1withhb + t2s[i]) / 2.
                addedhb = True
        if plot:
            ax = diag_plot(track, mass, ax=None, label='interp')
            if addedhb:
                ax = diag_plot(t1withhb, '', ax=ax, label='added hb')
                ax = diag_plot(t1hb, '', ax=ax, label='hb')

            ax = diag_plot(t1s[i], '', ax=ax, label='t1')
            ax = diag_plot(t2s[i], '', ax=ax, label='t2')
            plt.legend()
            figname = os.path.join(outdir, tname1s[i].replace('dat', 'png'))
            ax[1].set_xscale('log')
            plt.savefig(figname)
            print('wrote {}'.format(figname))
            plt.close()
        outfile = os.path.join(outdir, tname1s[i])
        fileio.savetxt(outfile, track, header=header, fmt='%.8f',
                       overwrite=overwrite)
        print('wrote {}'.format(outfile))

def rg_tip_heb_transition(hb_track, track):
    """
    Attach a HB model to a PMS model.
    Done in a consistent way as in TRILEGAL. Basically, zero time goes by,
    linear connection. The idea is that no stars should fall in this region
    because there are simply no tracks calculated. If you want a track following
    a hiashi line, well, calculate one. If you're interpolating, you're gonna
    have a slight error on a time scale of 1e5 years, counting a star that could
    have been in a transition phase from RG_TIP to HE_BEG as a RGB star.
    At this point in time, a negligable error.
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
    trans_track =  np.column_stack([logage, mass, logte, Mbol, logg, CO])

    hb_track.T[0] = np.log10(10 ** hb_track.T[0] + 10 ** logage[-1])
    new_track = np.concatenate((track, trans_track, hb_track))
    return new_track

def diag_plot(track, mass, ax=None, label=''):
    if ax is None:
        fig, ax = plt.subplots(ncols=2, figsize=(16,8))
        ax[0].set_xlabel(r'$\log\ Te$')
        ax[1].set_xlabel(r'$\rm{Age}$')
        ax[0].set_ylabel(r'$\rm{Mbol}$')

    ax[0].plot(track.T[2], track.T[3], label=label)
    if mass != '':
        ax[1].plot(10 ** track.T[0], track.T[3], label='${}\ {:.2f}$'.format(label, mass))
    else:
        ax[1].plot(10 ** track.T[0], track.T[3], label=label)

    return ax

def read_mhef(mhef_file):
    with open(mhef_file, 'r') as inp:
        lines = inp.readlines()
    zs = np.array([l.replace('Z', '') for l in lines[0].split() if 'Z' in l], dtype=float)
    data = np.genfromtxt(mhef_file)
    return data, zs

def interp_match_grid(parsecinterp_loc, mhef_file, overwrite=False,
                      plot=False):
    data, zs = read_mhef(mhef_file)
    subs = [l for l in os.listdir(parsecinterp_loc) if os.path.isdir(l)]
    pts = np.array([s.replace('ov', '') for s in subs], dtype=float)
    interps = [p for p in data.T[0] if not p in pts]
    newsubs=['ov{:.2f}'.format(s) for s in interps]
    sets = [[os.path.join(s, l) for l in os.listdir(s)] for s in subs]
    for i in range(len(sets)-1):
        for j in range(len(sets[i])):
            newdir = os.path.join(newsubs[i], os.path.split(sets[i][j])[1].replace('OV{:.1f}'.format(pts[i]), 'OV{:.2f}'.format(interps[i])))
            interpolate_between_sets(sets[i][j], sets[i+1][j], newdir, data[2*i+1][j+1],
                                     plot=plot)
    return
    """
    match_dirs1 = np.array(['MC_S13_OV0.3_Z0.002_Y0.2521',
                            'MC_S13_OV0.3_Z0.004_Y0.2557',
                            'MC_S13_OV0.3_Z0.008_Y0.2629'])#,
                            #'MC_S13_OV0.4_Z0.002_Y0.2521',
                            #'MC_S13_OV0.4_Z0.004_Y0.2557',
                            #'MC_S13_OV0.4_Z0.008_Y0.2629',
                            #'MC_S13_OV0.5_Z0.002_Y0.2521',
                            #'MC_S13_OV0.5_Z0.004_Y0.2557',
                            #'MC_S13_OV0.5_Z0.008_Y0.2629',
                            #'MC_S13_OV0.6_Z0.002_Y0.2521',
                            #'MC_S13_OV0.6_Z0.004_Y0.2557',
                            #'MC_S13_OV0.6_Z0.008_Y0.2629'])

    match_dirs2 = np.array(['MC_S13_OV0.4_Z0.002_Y0.2521',
                            'MC_S13_OV0.4_Z0.004_Y0.2557',
                            'MC_S13_OV0.4_Z0.008_Y0.2629'])#,
                            #'MC_S13_OV0.5_Z0.002_Y0.2521',
                            #'MC_S13_OV0.5_Z0.004_Y0.2557',
                            #'MC_S13_OV0.5_Z0.008_Y0.2629',
                            #'MC_S13_OV0.6_Z0.002_Y0.2521',
                            #'MC_S13_OV0.6_Z0.004_Y0.2557',
                            #'MC_S13_OV0.6_Z0.008_Y0.2629',
                            #'MC_S13_OV0.7_Z0.002_Y0.2521',
                            #'MC_S13_OV0.7_Z0.004_Y0.2557',
                            #'MC_S13_OV0.7_Z0.008_Y0.2629'])

    new_dirs = np.array(['MC_S13_OV0.35_Z0.002_Y0.2521',
                         'MC_S13_OV0.35_Z0.004_Y0.2557',
                         'MC_S13_OV0.35_Z0.008_Y0.2629'])#,
                         #'MC_S13_OV0.45_Z0.002_Y0.2521',
                         #'MC_S13_OV0.45_Z0.004_Y0.2557',
                         #'MC_S13_OV0.45_Z0.008_Y0.2629',
                         #'MC_S13_OV0.55_Z0.002_Y0.2521',
                         #'MC_S13_OV0.55_Z0.004_Y0.2557',
                         #'MC_S13_OV0.55_Z0.008_Y0.2629',
                         #'MC_S13_OV0.65_Z0.002_Y0.2521',
                         #'MC_S13_OV0.65_Z0.004_Y0.2557',
                         #'MC_S13_OV0.65_Z0.008_Y0.2629'])
    #v2
    #mhefs = np.array([1.82, 1.90, 1.95, 1.70, 1.80, 1.85, 1.62, 1.70, 1.75,
    #                  1.55, 1.60, 1.67])
    #v3
    mhefs = np.array([1.81, 1.86, 1.94])#, 1.70, 1.76, 1.84, 1.60, 1.66, 1.75,
    #                  1.51, 1.56, 1.66])

    for i in range(len(mhefs)):
        interpolate_between_sets(match_dirs1[i], match_dirs2[i], new_dirs[i],
                                 mhefs[i], overwrite=overwrite)
    """
def main(argv):
    """
    Main function for sfh.py plot sfh output from calcsfh, zcombine, or zcmerge
    """
    parser = argparse.ArgumentParser(description="Plot match sfh")

    parser.add_argument('-m', '--mhef_file', type=str,
                        help='file containing the He fusion masses')

    parser.add_argument('-i', '--isodir_loc', type=str,
                        help='where the isotrack files are (if not -m)')

    parser.add_argument('-p', '--parsecinterp_loc', type=str, default=os.getcwd(),
                        help='where the parsec for match files are')

    parser.add_argument('-d', '--diag_plot', type=str, default=os.getcwd(),
                        help='make HB plots')

    args = parser.parse_args(argv)

    #interp_mhefs()
    # then do this, but need to update mhefs

    # Where are the INT files
    if not args.mhef_file:
        isodir_loc = args.isodir_loc or os.getcwd()
        isodirs = [l for l in os.listdir(isodir_loc) if os.path.isdir(l)]
        args.mhef_file = interp_mhefs(isodirs)

    # Where are the match ready PARSEC files
    import pdb; pdb.set_trace()
    interp_match_grid(args.parsecinterp_loc, args.mhef_file, plot=args.diag_plot)

    # to do:
    #argparse it, make it read mhefs from file (at least?)

if __name__ == '__main__':
    main(sys.argv[1:])
