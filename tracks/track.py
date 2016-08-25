'''
padova track files
'''
from __future__ import print_function
import os
import sys

import numpy as np

from astropy.table import Table

from ..utils import get_zy, replace_
from ..eep.critical_point import CriticalPoint, Eep
from ..config import logL, logT, mass, age
from ..config import xcen, ycen, xc_cen, xo_cen, MODE, EXT


class AGBTrack(object):
    """
    AGBTrack adapted from colibri2trilegal
    """
    def __init__(self, filename):
        """
        Read in track, set mass and period.
        """
        self.match = False
        if 'match' in self.__dict__.keys() or 'match' in filename:
            self.match = True
        self.base, self.name = os.path.split(filename)

        if 'hb' in self.name.lower():
            self.hb = True
        self.load_agbtrack(filename)
        if not self.match:
            # period is either P0 or P1 based on value of Pmod
            period = np.zeros(len(self.data)) * np.nan
            for i, p in enumerate(self.data['Pmod']):
                if np.isfinite(p):
                    period[i] = self.data['P{:.0f}'.format(p)][i]
        try:
            self.mass = float(self.name.split('agb_')[1].split('_')[0])
        except:
            # self.mass = float(self.name.upper()
            #                   .split('M')[1].replace('.DAT', ''))
            self.mass = float('.'.join(self.name.upper()
                              .split('_M')[1].split('.')[:2]))
        try:
            self.filename_info()
        except:
            try:
                self.Z, self.Y = get_zy(self.base)
            except:
                self.Z = float(self.name.split('agb_')[1]
                               .split('_')[1].replace('Z', ''))
                self.Y = np.nan

    def ml_regimes(self):
        indi = None
        indf = None
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
        rdict = {u'#': u'', u'M_*': mass, u'lg ': u'log', u'_*': u'',
                 u'age/yr': age}
        with open(filename, 'r') as f:
            line = f.readline()
            names = replace_(line, rdict).strip().split()
            import pandas as pd
            self.data = pd.read_table(f, delim_whitespace=True, names=names)
        if not self.match:
            self.fix_phi()
        return self.data

    def fix_phi(self):
        '''The first line in the agb track is 1 but not a quiescent stage.'''
        istart = np.where(np.isfinite(self.data['NTP']))[0][0]
        self.data['PHI_TP'].iloc[istart] = np.nan

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

        if 'cstar' not in self.__dict__.keys():
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
        itpagb, = np.where(np.isfinite(self.data['NTP']))
        ntp = self.data['NTP'][itpagb]
        untp, itps = np.unique(ntp, return_index=True)
        itps += itpagb[0]
        if untp.size == 1:
            print('only one themal pulse.')
            self.tps = untp
        else:
            # The indices each TP is just filling values between the iTPs
            # and the final grid point
            itps = np.append(itps, len(ntp) + itpagb[0])
            tps = [np.arange(itps[i], itps[i+1]) for i in range(len(itps)-1)]
            self.tps = tps

    def get_quiescents(self):
        '''
        The quiescent phase, Qs,  is the the max phase in each TP,
        i.e., closest to 1.
        '''
        if 'tps' not in self.__dict__.keys():
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
        ycols = [logT, 'Tbot', logL, 'CO', mass, 'logdMdt']
        ylims = ylims or [None] * len(ycols)

        if axs is None:
            fig, axs = plt.subplots(nrows=len(ycols), sharex=True,
                                    figsize=(5.4, 10))
            fig.subplots_adjust(hspace=0.05, right=0.97, top=0.97, bottom=0.07,
                                left=0.2)

        for i in range(len(axs)):
            ycol = ycols[i]
            ylim = ylims[i]
            ax = axs[i]
            # ax.grid(ls='-', color='k', alpha=0.1, lw=0.5)
            ax.grid()
            try:
                ax.plot(self.data[age] / agescale, self.data[ycol], color='k')
            except:
                # period is not in the data but calculated in the init.
                ax.plot(self.data[age] / agescale,
                        self.__getattribute__(ycol), color='k')
            if ycol == 'CO':
                ax.axhline(1, linestyle='dashed', color='k', alpha=0.5, lw=1)
            ax.set_ylabel(translate_colkey(ycol), fontsize=20)
            ax.yaxis.set_major_locator(MaxNLocator(5, prune='upper'))
            if ylim is not None:
                # print ylim
                ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        axs[0].yaxis.set_major_locator(MaxNLocator(5, prune=None))
        ax.set_xlabel(translate_colkey(age, agescale=agescale), fontsize=20)
        [ax.get_yaxis().set_label_coords(-.16, 0.5) for ax in axs]
        # doesn't work with latex so well...
        axs[3].get_yaxis().set_label_coords(-.165, 0.5)
        [ax.get_yaxis().set_label_coords(-.17, 0.5)
         for ax in [axs[-1], axs[2]]]

        indi, indf = self.ml_regimes()
        if None not in [indi, indf]:
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


class Track(AGBTrack):
    '''Padova stellar track class.'''
    def __init__(self, filename, match=False, track_data=None,
                 ptcri_file=None, ptcri_kw=None, agb=False):
        '''
        filename [str] the path to the PMS or PMS.HB file
        '''
        (self.base, self.name) = os.path.split(filename)
        # will house error string(s)
        self.flag = None
        # will house interpolation messages
        self.info = {}

        self.hb = False
        if 'hb' in self.name.lower():
            self.hb = True

        self.agb = False
        if 'agb' in self.name.lower() or agb:
            self.agb = True
            AGBTrack.__init__(self, filename)

        self.match = match
        if self.match:
            self.load_match_track(filename, track_data=track_data)
        else:
            self.load_track(filename)

        # No errors so far
        if self.flag is None:
            # add self.Z etc.
            self.filename_info()
            self.track_mass()
            self.check_track()
            if ptcri_file is not None:
                self.load_iptcri(ptcri_file)

    def load_iptcri(self, ptcri_file):
        if isinstance(ptcri_file, str):
            ptcri = CriticalPoint(ptcri_file)
        else:
            ptcri = ptcri_file

        ptcri.load_iptcri(self)

    def check_track(self):
        '''check if age decreases'''
        try:
            age_ = np.round(self.data[age], 6)
        except AttributeError:
            age_ = np.round(self.data.logAge, 6)
        test = np.diff(age_) >= 0
        if False in test:
            # import pdb
            # pdb.set_trace()
            morp = 'parsec'
            if self.match:
                morp = 'match'

            print('decreasing age in {0:s} M={1:.3f} track'.format(morp,
                                                                   self.mass))
            bads, = np.nonzero(np.diff(age_) < 0)
            if not self.match:
                print('offensive {0:s}:'.format(MODE), self.data[MODE][bads])
                self.flag = 'track has age decreasing near MODEs {}' \
                            .format(self.data[MODE][bads])
            else:
                eep = Eep()
                if self.hb:
                    nticks = eep.nticks_hb
                    names = eep.eep_list_hb
                else:
                    nticks = eep.nticks
                    names = eep.eep_list
                inds = [np.argmin(np.abs(np.cumsum(nticks)-b)) for b in bads]
                print('offensive inds:', bads)
                print('Near:', np.array(names)[inds])
                # import pdb
                # pdb.set_trace()

        if not self.match:
            self.check_header_arg(loud=True)
            ycen_end = self.data[ycen][-1]
            if ycen_end != 0:
                self.info['Warning'] = \
                    'YCEN at final {0:s} {1:.4f}'.format(MODE, ycen_end)

        return

    def track_mass(self):
        ''' choose the mass based on the physical track starting points '''
        try:
            good_age, = np.nonzero(self.data[age] > 0.2)
        except AttributeError:
            # match tracks have log age
            good_age = [[0]]
        if len(good_age) == 0:
            self.flag = 'unfinished track'
            self.mass = self.data[mass][-1]
            return self.mass
        try:
            self.mass = self.data[mass][good_age[0]]
        except:
            e = sys.exc_info()
            print('Problem with Mass in {0}, {1}'.format(self.name, e))
            self.mass = self.data[mass][good_age[0]]
            import pdb
            pdb.set_trace()

        try:
            ind = self.name.lower().split('.').index('pms')
        except ValueError:
            ind = self.name.lower().split('.').index('dat')

        ext = self.name.split('.')[ind]

        ftmpmass = self.name.split('_M')[1].replace('.dat', '')
        npts = ftmpmass.count('.')
        while npts != 1:
            ftmpmass = ftmpmass[:-1]
            npts = ftmpmass.count('.')
        fmass = float(ftmpmass)
        if self.mass >= 12:
            self.mass = fmass
        elif np.abs(self.mass - fmass) > 0.005:
            print('filename has M=%.4f track has M=%.4f' % (fmass, self.mass))
            self.flag = 'inconsistent mass'
        return self.mass

    def calc_Mbol(self, z_sun=4.77):
        '''
        Uses Z_sun = 4.77 adds self.Mbol and returns Mbol
        '''
        self.Mbol = z_sun - 2.5 * self.data[logL]
        return self.Mbol

    def calc_logg(self):
        '''
        cgs constant is -10.616 adds self.logg and returns logg
        '''
        self.logg = -10.616 + np.log10(self.mass) + 4.0 * self.data[logT] - \
            self.data[logL]
        return self.logg

    def calc_core_mu(self):
        '''
        Uses X, Y, C, and O.
        '''
        xi = np.array([xcen, ycen, xc_cen, xo_cen])
        ai = np.array([1., 4., 12., 16.])
        # fully ionized
        qi = ai/2.
        try:
            self.muc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                    for i in range(len(xi))))
        except:
            xi = np.array([xcen, ycen, xc_cen, xo_cen])
            self.muc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                    for i in range(len(xi))))
        return self.muc

    def calc_lifetimes(self):
        self.tau_he = np.sum(self.data.Dtime[self.data['LY'] > 0])
        coreh, = np.nonzero((self.data['LX'] > 0) & (self.data[xcen] > 0))

        self.tau_h = np.sum(self.data.Dtime[coreh])
        return

    def filename_info(self):
        '''
        # get Z, Y into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        '''
        self.Z, self.Y = get_zy(self.name)

        if hasattr(self, 'header') and len(self.header) > 1:
            self.ALFOV, = \
                np.unique([float(l.replace('ALFOV', '').strip())
                           for l in self.header if ' ALFOV ' in l])

        if hasattr(self, 'data') and hasattr(self.data, 'QHEL'):
            if self.hb:
                self.zahb_mcore = self.data.QHEL[0]
            else:
                self.final_mcore = self.data.QHEL[-1]
        return

    def load_match_track(self, filename, track_data=None):
        '''
        load the match interpolated tracks into a record array.
        the file contains Mbol, but it is converted to LOG_L on read.
        LOG_L = (4.77 - Mbol) / 2.5
        names = 'logAge', 'mass', 'LOG_TE', 'LOG_L', 'logg', 'CO'
        '''
        def mbol2logl(m):
            try:
                logl = (4.77 - float(m)) / 2.5
            except TypeError:
                logl = (4.77 - m) / 2.5
            return logl

        self.col_keys = [age, mass, logT, logL, 'logg', 'CO']
        with open(filename, 'r') as inp:
            header = inp.readline()
            col_keys = header.split()
            if len(col_keys) > len(self.col_keys):
                self.col_keys.extend(col_keys[7:])

        if track_data is None:
            data = np.genfromtxt(filename, names=self.col_keys,
                                 converters={3: lambda m: mbol2logl(m)})
        else:
            track_data.T[3] = mbol2logl(track_data.T[3])
            dtype = [(c, float) for c in self.col_keys]
            nrows = len(track_data)
            data = np.ndarray(shape=(nrows,), dtype=dtype)
            for i in range(nrows):
                data[i] = track_data[i]

        eep = Eep()
        iptcri = np.cumsum(eep.nticks) - 1.
        iptcri[iptcri >= len(data)] = 0
        self.iptcri = np.array(iptcri, dtype=int)
        self.data = data.view(np.recarray)
        self.mass = self.data[mass][0]
        return data

    def load_track(self, filename):
        '''
        reads PMS file into a record array. Stores header and footer as
        list of strings in self.header

        if no 'BEGIN TRACK' in file will add message to self.flags

        if no footers are found will add message to self.info
        '''
        if 'ALFO0' in filename:
            print('Warning: loading an ALFO0 track. {}'.format(filename))

        with open(filename, 'r') as infile:
            lines = infile.readlines()

        rdict = {'LOG_L': logL, 'LOG_TE': logT, 'AGE': age, 'MASS': mass}
        # find the header
        begin_track = -1
        if 'BEGIN TRACK' in ''.join(lines):
            for i, l in enumerate(lines):
                if 'BEGIN TRACK' in l:
                    begin_track = i
                    break

        header = ['']
        if begin_track > -1:
            header = lines[:begin_track]
            if type(header) is not list:
                header = [header]
            begin_track += 1

        self.header = header

        if begin_track == -1:
            try:
                with open(filename) as inp:
                    names = replace_(inp.readline().strip(), rdict).split()

                self.data = np.array(Table.read(filename, format='ascii',
                                                names=names)).view(np.recarray)
                self.col_keys = np.array(self.data.dtype.names)
            except:
                self.data = np.array([])
                self.col_keys = None
                self.flag = 'load_track error: no begin track '
            self.mass = \
                float('.'.join(self.name.split('_M')[1].split('.')[:2]))
            return

        if len(lines) - begin_track <= 2:
            self.data = np.array([])
            self.col_keys = None
            self.flag = 'load_track error: no data after begin track'
            self.mass = \
                float(self.name.split('_M')[1]
                      .replace('.DAT', '').replace('.PMS', '').split('.HB')[0])
            return

        # find the footer assuming it's no longer than 5 lines (for speed)
        # (the footer will not start with the integer MODEL number)
        skip_footer = 0
        for l in lines[-5:]:
            try:
                int(l.split()[0])
            except ValueError:
                skip_footer -= 1

        self.header.extend([' # Footer: %s lines \n' % skip_footer])
        if skip_footer < 0:
            self.header.extend(lines[skip_footer:])
        else:
            self.info['load_track warning'] = \
                'No footer unfinished track? %s' % filename

        # find ndarray titles (column keys)

        col_keys = replace_(lines[begin_track].replace('#', '').strip(),
                            rdict).split()
        begin_track += 1

        # extra line for tracks that have been "colored"
        if 'information' in lines[begin_track + 1]:
            begin_track += 1
            col_keys = self.add_to_col_keys(col_keys, lines[begin_track])

        # read data into recarray
        iend = len(lines) + skip_footer
        nrows = iend - begin_track
        dtype = [(c, float) for c in col_keys]

        data = np.ndarray(shape=(nrows,), dtype=dtype)
        for row, i in enumerate(range(begin_track, iend)):
            try:
                data[row] = tuple(lines[i].split())
            except:
                e = sys.exc_info()[0]
                print('Problem with line {0} of {1}, {2}'
                      .format(i, filename, e))
                break

        self.data = data.view(np.recarray)
        self.col_keys = col_keys
        return

    def summary(self, line=''):
        """
        make a summary line of the track

        track_type Z mass ALFOV QHEL tau_He tau_H
        if self.HB, track_type = 0, QHEL is at the ZAHB
        otherwise,  track_type = 1, QHEL is the final track point.
        tau_* is the core fusion lifetimes of *.

        if there is a self.flag, will add a comment line with the flag

        Parameters
        ----------
        line : string default: ''
            string to add the summary to

        Returns
        -------
        line : string
            a line with format: track_type Z mass ALFOV QHEL tau_He tau_H
            or a comment: self.Z self.name self.flag
        """
        if self.flag is None:
            fmt = '%i %g %5.3f %.2f %.3f %.4g %.4g'
            self.calc_lifetimes()
            if self.hb:
                line += fmt % (1, self.Z, self.mass, self.ALFOV,
                               self.zahb_mcore, self.tau_he, 0.)
            else:
                line += fmt % (0, self.Z, self.mass, self.ALFOV,
                               self.final_mcore, self.tau_he, self.tau_h)
        else:
            efmt = '# %.3f %s: %s \n'
            print(efmt % (self.Z, self.name, self.flag))
        return line

    def add_to_col_keys(self, col_keys, additional_col_line):
        '''
        If fromHR2mags was run, A new line "Additional information Added:..."
        is added, this adds these column keys to the list.
        '''
        new_cols = additional_col_line.split(':')[1].strip().split()
        col_keys = list(np.concatenate((col_keys, new_cols)))
        return col_keys

    def add_header_args_dict(self):
        """PARSEC header as a dictionary"""
        def rep(s):
            return s.replace('.D', '.E')

        def parse_args(header_line):
            header_line = header_line.replace('*', '')
            k, v = zip(*[a.split('=') for a in
                       [l for l in header_line.split()
                        if '=' in l and 'RESTART' not in l]])
            arg_dict = dict(zip(k, map(float, map(rep, v))))
            return arg_dict

        def update_args(header_line, old_dict):
            old_dict.update(parse_args(header_line))
            return old_dict

        self.header_dict = {}
        for line in self.header:
            if line.count('=') > 1:
                self.header_dict = update_args(line, self.header_dict)

    def check_header_arg(self, ok_eval='%f>1.3e10', arg='AGELIMIT',
                         errstr='AGE EXCEEDS AGELIMIT', loud=False):
        """
        To save computational time, tracks can be calculated with an
        artificial age limit. However, a fully populated age grid is needed
        in match.
        """
        if not hasattr(self, 'header_dict'):
            self.add_header_args_dict()
        check = len([i for i, l in enumerate(self.header) if errstr in l])
        if check > 0:
            if ok_eval % self.header_dict[arg]:
                level = 'warning'
            else:
                level = 'error'
            msg = '%s %s: %g' % (level, errstr, self.header_dict[arg])
            if loud:
                print(msg)
            self.info['%s %s' % (level, arg)] = \
                '%s: %g' % (errstr, self.header_dict[arg])


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
             logT: r'$log\ \rm{T}_{\rm{eff}}\ \rm{(K)}$',
             logL: r'$log\ L\ (L_\odot)$',
             'period': r'$\rm{P\ (days)}$',
             'CO': r'$\rm{C/O}$',
             mass: r'$\rm{M}\ (\rm{M}_\odot)$',
             'logdMdt': r'$\dot{\rm{M}}\ (\rm{M}_\odot/\rm{yr})$',
             age: r'$\rm{TP-AGB\ Age\ (%syr)}$' % str_agescale(agescale)}

    new_col = col
    if col in tdict.keys():
        new_col = tdict[col]

    return new_col


if __name__ == "__main__":
    line = 'HB Z M OV QHEL tau_He tau_H\n'
    for tn in sys.argv[1:]:
        t = Track(tn)
        if t.flag is None:
            line += '{0:s}\n'.format(t.summary())
        else:
            print(t.mass, t.flag)

    with open('track_summary.dat', 'w') as outp:
        outp.write(line)
