import os
import numpy as np
from ...fileio.fileIO import get_files
from astroML.stats import binned_statistic_2d
import matplotlib.pylab as plt

class MatchTrack(object):
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.parse_filename()
        self.load_data(filename)

    def parse_filename(self):
        """
        set attributes from the filename, converts from MATCH's [Z] - 4. to Z
        """
        items = self.name.split('_')
        self.mod = items[0]
        keys = ['lagei', 'lagef', 'logz', 'dlogz']
        for key, val in zip(keys, items[1:]):
            if 'logz' in key:
                val = 0.02 * 10 ** (float(val) - 4.)
            self.__setattr__(key, float(val))

    def load_data(self, fname):
        """ load MATCH track, convert Mbol and ln Mass to LOG_L and Mass """
        keys = ['LOG_L', 'LOG_TE', 'Nstars', 'MASS', 'LOGG']
        self.data = np.genfromtxt(fname, names=keys,
                                  converters={0: lambda x: (4.77 - float(x)) / 2.5,
                                              3: lambda x: np.exp(float(x))})
        return

    def get_val(self, attr):
        """ get the value of an attribute in self.data """
        if type(attr) == str:
            val = self.data[attr]
        else:
            val = attr
        return val

    def bs2d(self, x, y, z, stat='count', bins='default'):
        """
        call astroML.binned_statistic_2d
        if bins == 'default' will send unique values of x and y as bins
        see astroML.binned_statistic_2d doc

        Parameters
        ----------
        x, y, z : array or string
            arrays or names of data arrays to send to binned_statistic_2d
        """
        x = self.get_val(x)
        y = self.get_val(y)
        z = self.get_val(z)

        if bins == 'default':
            bins = [np.unique(x), np.unique(y)]
        return binned_statistic_2d(x, y, z, stat, bins=bins)

    def hrd(self, z='Nstars', bins=200):
        ax1 = plt.subplot2grid((3,3), (1,0), rowspan=2, colspan=2)
        ax2 = plt.subplot2grid((3,3), (0,0), colspan=2)
        ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
        l = ax1.scatter(data1['LOG_TE'], data1['LOG_L'], c=data1['MASS'],
                        edgecolors='none', cmap=plt.cm.Blues)

        if z != 'Nstars':
            N, xe, ye = self.bs2d('LOG_TE', 'LOG_L', z, bins=bins)
            l = ax.imshow(N.T, extent=[xe[-1], xe[0], ye[-1], ye[0]],
                           cmap=plt.cm.Blues, interpolation='nearest', aspect=0.1)
            cb = plt.colorbar(l)
        else:
            pass

        ax.set_xlabel(r'$\log\ T_{\rm{eff}}\ \rm{(K)}$', fontsize=20)
        ax.set_ylabel(r'$\log\ L\ {\rm{(L_\odot)}}$', fontsize=20)
        ax.set_title(r'$Z=%.4f\ \log {\rm{Age:}} %g-%g$' % \
                     (self.logz, self.lagei, self.lagef), fontsize=12)
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])
        cb.set_label('$%s$' % z, fontsize=16)
        return ax

class MatchTracks(object):
    def __init__(self, base, binaries=False):
        self.base = base
        if binaries:
            search_term = 'mod2_*'
        else:
            search_term = 'mod1_*'
        filenames = get_files(base, search_term)
        self.tracks = [MatchTrack(f) for f in filenames]

    def plot_hrds(self, outfile_dir=None):
        if outfile_dir is None:
            outfile_dir = self.base
        for t in self.tracks:
            fname = os.path.join(outfile_dir, t.name + '.png')
            t.hrd()
            plt.savefig(fname)
            plt.close()
