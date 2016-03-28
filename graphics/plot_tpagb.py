import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from matplotlib.ticker import MaxNLocator
from palettable.wesanderson import Darjeeling2_5


from ..utils import minmax, replace_
from ..tracks.tracks import AGBTrack

sns.set()
sns.set_context('paper')
plt.style.use('paper')


def duration_masslost(agbs, justprint=False, norm=False):
    if justprint:
        aa = [3., 4., 5.]
        for a in aa:
            for agb in agbs:
                if agb.Z not in [0.001, 0.008]:
                    continue
                #plt.plot(agbs[i].data[age], agbs[i].data['L_star'])
                ind1, ind2 = agb.ml_regimes()
                if not None in [ind1, ind2] or ind1 != ind2:
                    if agb.mass != a:
                        continue
                    age = agb.data[age] / 1e5
                    mass = agb.data[mass]
                    #print sum(agb.data['dt'][np.nonzero(agb.data['L_star'] < 3.4)[0]])
                    print(ind1, ind2)
                    print '{:g} {:g} {:.2f} {:.2f} {:.2f} {:.2f}'.format(agb.Z, agb.mass,
                            age[ind1], age[ind2]-age[ind1], age[-1]-age[ind2], age[-1])

    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8, 4))
    #sns.despine()
    col1, col2 = axs.T[0], axs.T[1]
    colors = Darjeeling2_5.mpl_colors[1:-1]
    kw = {'align':'edge'}
    for agb in agbs:
        if agb.mass >= 3.2 and agb.mass <= 5.:
            for i, col in enumerate([col1, col2]):
                if agb.Z == 0.001:
                    ax = col[0]
                if agb.Z == 0.004:
                    continue
                if agb.Z == 0.008:
                    ax = col[1]
                ipd, = np.nonzero(agb.data['M_predust'] == agb.data['dMdt'])
                idd, = np.nonzero(agb.data['Mdust'] == agb.data['dMdt'])
                iall = np.arange(len(agb.data['dMdt']))
                isw = np.array(list(set(iall) - set(ipd) - set(idd)))
                ttp = 1e5
                if norm:
                    ttp = np.sum(agb.data['dt'])
                tpd = np.sum(agb.data['dt'][ipd]) / ttp
                tdd = np.sum(agb.data['dt'][idd]) / ttp
                tsw = np.sum(agb.data['dt'][isw]) / ttp
                if agb.Z == 0.001 and agb.mass == 5. and i == 0:
                    ax.barh(agb.mass, tpd, 0.2, color=colors[0], label=r'$\dot{M}_{pd}$', **kw)
                    ax.barh(agb.mass, tdd, 0.2, color=colors[1], label=r'$\dot{M}_{dd}$', left=tpd, **kw)
                    ax.barh(agb.mass, tsw, 0.2, color=colors[2], label=r'$\dot{M}_{sw}$', left=tdd+tpd, **kw)
                    if norm:
                        loc = 'upper left'
                        frameon = True
                    else:
                        loc='upper right'
                        frameon = False
                    ax.legend(labelspacing=0.02, loc=loc, frameon=frameon, fontsize=10,  handlelength=1)
                else:
                    if i > 0:
                        ttp = 1
                        if norm:
                            ttp = agb.mass
                        tpd = np.sum(agb.data['dt'][ipd] * agb.data['dMlost'][ipd]) / ttp
                        tdd = np.sum(agb.data['dt'][idd] * agb.data['dMlost'][idd]) / ttp
                        tsw = np.sum(agb.data['dt'][isw] * agb.data['dMlost'][isw]) / ttp
                        if norm:
                            ax.set_xlim(0, 1)
                        if agb.mass == 5.:
                            ax.text(0.98, 0.02, r'$\rm{Z}=%g$' % agb.Z, fontsize=16,
                                    transform=ax.transAxes, ha='right')

                    ax.barh(agb.mass, tpd, 0.2, color=colors[0], **kw)
                    ax.barh(agb.mass, tdd, 0.2, color=colors[1], left=tpd, **kw)
                    ax.barh(agb.mass, tsw, 0.2, color=colors[2], left=tdd+tpd, **kw)

    for ax in axs.flatten():
        ax.tick_params(direction='out', color='k', size=2.6, width=0.5)
        #ax.grid(lw=0.6, color='k')
        ax.grid()
        if not norm:
            ax.set_xlim(ax.set_xlim(0, 4.5))
        ax.set_ylim(3.2, 5.2)

    [ax.tick_params(labelbottom='off') for ax in axs.flatten()[:-2]]
    [ax.tick_params(labelright='on') for ax in col2]
    N = len(col2[-1].get_xticks())
    [ax.xaxis.set_major_locator(MaxNLocator(N, prune='lower')) for ax in col2]
    if norm:
        col1[-1].set_xlabel(r'$\rm{TP-AGB\ Lifetime}$')
        col2[-1].set_xlabel(r'$\rm{Fraction\ of\ Initial\ Mass\ Lost}$')
        [ax.set_xlim(0, 0.9) for ax in col2]
        [ax.set_xlim(0, 1) for ax in col1]
    else:
        col1[-1].set_xlabel(r'$\rm{TP-AGB\ Age\ (10^5\ yr)}$')
        col2[-1].set_xlabel(r'$\rm{Mass\ Lost\ (M_\odot)}$')

    fig.text(0.03, 0.58, r'$\rm{TP-AGB\ Initial\ Mass (M_\odot)}$', rotation='vertical', ha='center', va='center',
             fontsize=20)
    fig.subplots_adjust(left=0.1, hspace=0.05, wspace=0.05, right=0.92, bottom=0.2, top=0.98)
    if norm:
        plt.savefig('duration_masslost_norm{}'.format(EXT))
    else:
        plt.savefig('duration_masslost{}'.format(EXT))
    return fig, axs


def compare_vw93(agbs, outfile=None, xlim=None, ylims=None):
    if agbs[0].mass != agbs[1].mass:
        annotations = [None, None]
    else:
        fmt = r'$\rm{Z}=%g$'
        annotations = [fmt % agbs[0].Z, fmt % agbs[1].Z]
    # sharex is off because I want one column pruned.
    fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(9, 10))
    col1 = axs.T[0]
    col2 = axs.T[1]

    fig, col1 = agbs[0].vw93_plot(fig=fig, axs=col1, annotation=annotations[0])
    fig, col2 = agbs[1].vw93_plot(fig=fig, axs=col2, annotation=annotations[1])

    for i in range(len(col1)):
        ax1 = col1[i]
        ax2 = col2[i]
        [ax.set_ylim(minmax(ax1.get_ylim(), ax2.get_ylim())) for ax in [ax1, ax2]]
        [ax.set_xlim(minmax(ax1.get_xlim(), ax2.get_xlim())) for ax in [ax1, ax2]]
        ax2.tick_params(labelleft=False, labelright=True)
        [ax.tick_params(labelbottom=False) for ax in [ax1, ax2]]
        ax2.set_ylabel('')

    [ax.tick_params(labelbottom=True) for ax in [col1[-1], col2[-1]]]
    # prune only the left column
    N = len(col1[0].get_xticks())
    col1[-1].xaxis.set_major_locator(MaxNLocator(N, prune='upper'))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.02)
    if outfile is not None:
        plt.savefig(outfile)
        print('wrote {}'.format(outfile))
    return fig, (col1, col2)

def main(argv):
    parser = argparse.ArgumentParser(description="Make a plot like Vassiliadis and Wood 1993")

    parser.add_argument('infiles', type=str, nargs='*',
                        help='COLIBRI track file(s)')

    parser.add_argument('-x', '--xlim', type=str, default=None,
                        help='comma separated x axis limits')

    parser.add_argument('-t', '--tlim', type=str, default=None,
                        help='comma separated log teff axis limits')

    parser.add_argument('-l', '--llim', type=str, default=None,
                        help='comma separated log l axis limits')

    parser.add_argument('-p', '--plim', type=str, default=None,
                        help='comma separated period axis limits')

    parser.add_argument('-c', '--clim', type=str, default=None,
                        help='comma separated c/o axis limits')

    parser.add_argument('-m', '--mlim', type=str, default=None,
                        help='comma separated mass axis limits')

    parser.add_argument('-d', '--dmlim', type=str, default=None,
                        help='comma separated mass loss axis limits')

    parser.add_argument('-z', '--compare', action='store_true',
                        help='one plot')

    parser.add_argument('-f', '--dmplot', action='store_true',
                        help='duration mass lost plot')

    parser.add_argument('-n', '--norm', action='store_false',
                        help='with -f do not norm (use units)')

    args = parser.parse_args(argv)

    ylims = [args.tlim, args.llim, args.plim, args.clim, args.mlim, args.dmlim]
    xlim = args.xlim

    if args.dmplot:
        agbs = [AGBTrack(infile) for infile in args.infiles]
        duration_masslost(agbs, norm=args.norm)

    elif args.compare:
        agbs = [AGBTrack(infile) for infile in args.infiles]
        outfile = 'tpagb_comp'
        if agbs[0].mass == agbs[1].mass:
            outfile += '_m%g' % agbs[0].mass
        elif agbs[0].Z == agbs[1].Z:
            outfile += '_z%g' % agbs[0].mass
        outfile += EXT

        compare_vw93(agbs, outfile=outfile, xlim=xlim, ylims=ylims)
    else:
        for infile in args.infiles:
            agb = AGBTrack(infile)
            outfile = infile.replace('.dat', EXT)
            agb.vw93_plot(outfile=outfile, xlim=xlim, ylims=ylims)
            plt.close()

def default_run():
    print('python -m tpagb_calibration.plotting.plot_tracks -f ~/research/TP-AGBcalib/AGBTracks/CAF09/S_NOV13/S12_Z0.001_Y0.250/agb_*Mdot50*.dat ~/research/TP-AGBcalib/AGBTracks/CAF09/S_NOV13/S12_Z0.008_Y0.263/agb_*Mdot50*.dat')


if __name__ == "__main__":
    main(sys.argv[1:])
