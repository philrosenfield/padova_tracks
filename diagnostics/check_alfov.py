import argparse
import matplotlib.pylab as plt
import numpy as np
import seaborn
seaborn.set()


def plot_alfov(alfov_report):
    """
    Plot ALFOV vs mass from a file containing PARSEC parameter calls.
    """
    alfov = open(alfov_report, 'r').readlines()

    fnames = np.array([line.split(':')[0] for line in alfov])
    # ALFOV in the parameter call this doesn't work for grep ALFOV
    # because there is another line in the header:
    # "ALFOV       [some number]"
    alf = np.array([line.split('ALFOV=')[1].split()[0]
                    for line in alfov], dtype=float)
    # ALFOV in the filename. Assuming format: *OV0.3_*
    ovfile = np.array([fname.split('OV')[0].split('_')[0],
                       for fname in fnames], dtype=float)
    # Assuming format: *M1.30.P*
    mass = np.array([fname.split('M')[1].split('.P')[0],
                     for fname in fnames], dtype=float)

    uinds, uarr = np.unique(ovfile, return_index=True)
    uarr = np.append(uarr, len(fnames) - 1)

    fig, ax = plt.subplots()
    for i in range(len(uarr) - 1):
        inds = np.arange(uarr[i], uarr[i+1])
        smass = np.sort(mass[inds])
        salf = alf[inds][np.argsort(mass[inds])]
        l, = ax.plot(smass, salf, label=np.unique(ovfile[inds]))
        ax.plot(smass, salf, 'o', color=l.get_color())

    plt.legend(loc='best')
    ax.set_ylabel(r'$\rm{ALFOV}$')
    ax.set_xlabel(r'$\rm{M}\ (M_\odot)$')
    ax.set_ylim(0, max(alf) + 0.1)
    ax.set_xlim(0., 2)
    plt.savefig('alfov_mass.png')
    return ax


def main(args):
    parser = argparse.ArgumentParser(description="Plot ALFOV vs Mass")

    parser.add_argument('alfov_report', type=str,
                        help=('ALFOV report e.g., ',
                              'grep "ALFOV=" */*F7_*.PMS > alfov_report.txt'))

    args = parser.parse_args(argv)
    plot_alfov(args.alfov_report)

if __name__ == '__main__':
    main(sys.argv[1:])
