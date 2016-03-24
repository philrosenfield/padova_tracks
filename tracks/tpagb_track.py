import numpy as np
from ..utils import replace_

def translate_colkey(col, agescale=1.):
    """
    Turn COLIBRI column name into a axes label
    """
    def str_agescale(scale=1.):
        """
        Set the age unit string.
        """
        u = ''
        if scale == 1e9:
            u = 'G'
        elif scale == 1e6:
            u = 'M'
        elif np.log10(scale) >= 1.:
            u = '10^%i\ ' % int(np.log10(scale))
        return u

    tdict = {'Tbot': r'$log\ \rm{T}_{\rm{bce}}\ \rm{(K)}$',
             'logT': r'$log\ \rm{T}_{\rm{eff}}\ \rm{(K)}$',
             'logL': r'$log\ L\ (L_\odot)$',
             'period': r'$\rm{P\ (days)}$',
             'CO': r'$\rm{C/O}$',
             'mass': r'$\rm{M}\ (\rm{M}_\odot)$',
             'logdMdt': r'$\dot{\rm{M}}\ (\rm{M}_\odot/\rm{yr})$',
             'age': r'$\rm{TP-AGB\ Age\ (%syr)}$' % str_agescale(agescale)}

    new_col = col
    if col in tdict.keys():
        new_col = tdict[col]

    return new_col

class AGBInp(object):
    """
    Input files between PARSEC and COLIBRI
    """
    def __init__(self, filename):
        self.load_inp(filename)

    def load_inp(self, filename):
        rdict = {'#': '', 'm1': 'mass', ' l1 ': 'logL', ' te1 ': 'logT'}
        with open(filename, 'r') as inp:
            names = replace_(inp.readline().strip(), rdict).split()
            self.data = np.genfromtxt(filename, names=names)

# last line??
# where is Age??



class AGBTrack(object):
    """
    AGBTrack adapted from colibri2trilegal
    """
    def __init__(self, filename):
        """
        Read in track, set mass and period.
        """
        self.load_agbtrack(filename)
        # period is either P0 or P1 based on value of Pmod
        self.period = np.array([self.data['P{:.0f}'.format(p)][i]
                                for i, p in enumerate(self.data['Pmod'])])
        self.mass = float(filename.split('agb_')[1].split('_')[0])
        self.Z = float(filename.split('agb_')[1].split('_')[1].replace('Z',''))

    def ml_regimes(self):
        indi=None
        indf=None
        try:
            arr, = np.nonzero(self.data['Mdust'] == self.data['dMdt'])
            indi = arr[0]
            indf = arr[-1]
        except:
            pass
        return indi, indf

    def load_agbtrack(self, filename):
        '''
        Load COLIRBI track and make substitutions to column headings.
        '''
        rdict = {'#': '', 'M_*': 'mass', 'lg ': 'log', '_*': '', 'age/yr': 'age'}
        with open(filename, 'r') as f:
            line = f.readline()
            self.data = np.genfromtxt(f, names=replace_(line, rdict).strip().split())
        self.fix_phi()
        return self.data


    def fix_phi(self):
        '''The first line in the agb track is 1. This isn't a quiescent stage.'''
        self.data['PHI_TP'][0] = np.nan

    def m_cstars(self, mdot_cond=-5, logl_cond=3.3):
        '''
        adds mstar and cstar attribute of indices that are true for:
        mstar: co <=1 logl >= 3.3 mdot <= -5
        cstar: co >=1 mdot <= -5
        (by default) adjust mdot with mdot_cond and logl with logl_cond.
        '''
        data = self.data_array

        self.mstar, = np.nonzero((data['CO'] <= 1) &
                                 (data['logL'] >= logl_cond) &
                                 (data['logdMdt'] <= mdot_cond))
        self.cstar, = np.nonzero((data['CO'] >= 1) &
                                 (data['logdMdt'] <= mdot_cond))

    def tauc_m(self):
        '''lifetimes of c and m stars'''

        if not 'cstar' in self.__dict__.keys():
            self.m_cstars()
        try:
            tauc = np.sum(self.data['dt'][self.cstar]) / 1e6
        except IndexError:
            tauc = np.nan
        try:
            taum = np.sum(self.data['dt'][self.mstar]) / 1e6
        except IndexError:
            taum = np.nan
        self.taum = taum
        self.tauc = tauc

    def get_tps(self):
        '''find the thermal pulsations of each file'''
        self.tps = []
        ntp = self.data['NTP']
        untp, itps = np.unique(ntp, return_index=True)
        if untp.size == 1:
            print('only one themal pulse.')
            self.tps = untp
        else:
            # The indices each TP is just filling values between the iTPs
            # and the final grid point
            itps = np.append(itps, len(ntp))
            self.tps = [np.arange(itps[i], itps[i+1])
                        for i in range(len(itps) - 1)]

    def get_quiescents(self):
        '''
        The quiescent phase, Qs,  is the the max phase in each TP,
        i.e., closest to 1.
        '''
        if not 'tps' in self.__dict__.keys():
            self.get_tps()
        phi = self.data['PHI_TP']
        logl = self.data['logL']
        self.iqs = np.unique([tp[np.argmax(phi[tp])] for tp in self.tps])
        self.imins = np.unique([tp[np.argmin(logl[tp])] for tp in self.tps])


    def vw93_plot(self, agescale=1e5, outfile=None, xlim=None, ylims=None,
                  fig=None, axs=None, annotate=True, annotation=None):
        """
        Make a plot similar to Vassiliadis and Wood 1993. Instead of Vesc,
        I plot C/O.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from palettable.wesanderson import Darjeeling2_5
        sns.set()
        sns.set_context('paper')
        plt.style.use('paper')

        age = 'age'
        ycols = ['T_star', 'Tbot', 'L_star', 'CO', 'M_star', 'dMdt']
        ylims = ylims or [None] * len(ycols)

        if axs is None:
            fig, axs = plt.subplots(nrows=len(ycols), sharex=True, figsize=(5.4, 10))
            fig.subplots_adjust(hspace=0.05, right=0.97, top=0.97, bottom=0.07,
                                left=0.2)

        for i in range(len(axs)):
            ycol = ycols[i]
            ylim = ylims[i]
            ax = axs[i]
            #ax.grid(ls='-', color='k', alpha=0.1, lw=0.5)
            ax.grid()
            try:
                ax.plot(self.data[age] / agescale, self.data[ycol], color='k')
            except:
                # period is not in the data but calculated in the init.
                ax.plot(self.data[age] / agescale, self.__getattribute__(ycol), color='k')
            if ycol == 'CO':
                ax.axhline(1, linestyle='dashed', color='k', alpha=0.5, lw=1)
            ax.set_ylabel(translate_colkey(ycol), fontsize=20)
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            if ylim is not None:
                print ylim
                ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        axs[0].yaxis.set_major_locator(MaxNLocator(5, prune=None))
        ax.set_xlabel(translate_colkey(age, agescale=agescale), fontsize=20)
        [ax.get_yaxis().set_label_coords(-.16,0.5) for ax in axs]
        # doesn't work with latex so well...
        axs[3].get_yaxis().set_label_coords(-.165,0.5)
        [ax.get_yaxis().set_label_coords(-.17,0.5) for ax in [axs[-1], axs[2]]]

        indi, indf = self.ml_regimes()
        if not None in [indi, indf]:
            [[ax.axvline(self.data[age][i]/agescale, ls=':', color='k',
                         alpha=0.5, lw=0.8)
            for ax in axs] for i in [indi, indf]]
        if annotate:
            if annotation is None:
                annotation = r'$\rm{M}_i=%.2f\ \rm{M}_\odot$' % self.mass
            axs[4].text(0.02, 0.05, annotation, ha='left', fontsize=16,
                        transform=axs[4].transAxes)

        if outfile is not None:
            plt.tight_layout()
            plt.savefig(outfile)
        return fig, axs

