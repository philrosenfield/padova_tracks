import matplotlib.pylab as plt
import numpy as np
import os
import trilegal
from ResolvedStellarPops import fileio

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

def interp_mhefs(isotracks=None, outfile=None):
    """
    Write the minimum initial mass for He fusion to a file, interpolating
    between isotracks.

    Parameters
    ----------
    isotracks : list of strings
        path to parsec isotrack files. Must be in order!
        eg:
        isotracks = ['isotrack/parsec/CAF09_MC_S13v3_OV0.3.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.4.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.5.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.6.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.7.dat']
    outfile : string
        filename of output file
    """
    def pretty(ov, marr):
        """ make a %.2f string combining a float and an array """
        return ' '.join(['%.2f' % i for i in np.concatenate(([ov], marr))]) + '\n'

    if isotracks is None:
        isotracks = ['isotrack/parsec/CAF09_MC_S13v3_OV0.3.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.4.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.5.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.6.dat',
                     'isotrack/parsec/CAF09_MC_S13v3_OV0.7.dat']
        isotracks = [os.path.join(os.environ['TRILEGAL_ROOT'], i)
                     for i in isotracks]

    if outfile is None:
        outfile = os.path.join(os.environ['TRILEGAL_ROOT'],
                               'CAF09_MC_S13v3_MHeF_interp.dat')
    line = '\n'.join(['# %s' % i for i in isotracks]) + '\n'
    isots = [trilegal.IsoTrack(i) for i in isotracks]
    line += '# OV '
    for isot in isots:
        isot.load_int2()
        isot.ov = float(isot.name.split('OV')[1].replace('.dat', ''))
    line += ' '.join(['%.3f' % z for z in isot.Z]) + '\n'
    for i in range(len(isots)-1):
        intov = (isots[i+1].ov + isots[i].ov) / 2.
        intpd = (np.array(isots[i+1].mhefs) + np.array(isots[i].mhefs)) / 2.
        line += pretty(isots[i].ov, isots[i].mhefs)
        line += pretty(intov, intpd)
    line += pretty(isots[i+1].ov, isots[i+1].mhefs)

    with open(outfile, 'w') as outf:
        outf.write(line)
    print 'wrote %s' % outfile

def interpolate_between_sets(match_dir1, match_dir2, outdir, mhef,
                             overwrite=False):
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
        import pdb; pdb.set_trace()

    t1s = [np.loadtxt(t) for t in t1files]
    t2s = [np.loadtxt(t) for t in t2files]
    for i in range(ntracks):
        mass = strip_m(t1files[i])
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
                t1withhb = rg_tip_heb_transition(t1hb, t1s[i])
                track = (t1withhb + t2s[i]) / 2.
                #ax = diag_plot(track, mass, ax=ax, color='k')
                #ax = diag_plot(t1withhb, '', ax=ax, color='r')
                #ax = diag_plot(t1s[i], '', ax=ax, color='r')
                #ax = diag_plot(t2s[i], '', ax=ax, color='b')
        outfile = os.path.join(outdir, tname1s[i])
        fileio.savetxt(outfile, track, header=header, fmt='%.8f',
                       overwrite=overwrite)

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
    # hard coded! whatever, you can get it from eep.critical_points.critical_point

    ntrans = 40
    rg_tip = 1468

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

def diag_plot(track, mass, ax=None, color='k'):
    if ax is None:
        fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8,8))
        ax[0].set_xlabel(r'$\log\ Te$')
        ax[1].set_xlabel(r'$\rm{Age}$')
        ax[0].set_ylabel(r'$\rm{Mbol}$')

    ax[0].plot(track.T[2], track.T[3], color=color)
    if mass != '':
        ax[1].plot(10 ** track.T[0], track.T[3], label='$%.2f$' % mass,
                   color=color)
    else:
        ax[1].plot(10 ** track.T[0], track.T[3], color=color)

    return ax



def interp_match_grid(overwrite=False):
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

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    #with new version, first do this:
    #interp_mhefs()
    # then do this, but need to update mhefs
    interp_match_grid()

    # to do:
    #argparse it, make it read mhefs from file (at least?)
