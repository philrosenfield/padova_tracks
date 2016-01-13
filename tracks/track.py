'''
padova track files
'''
from __future__ import print_function
import os
import sys

import numpy as np

class Track(object):
    '''
    Padova stellar track.
    '''
    def __init__(self, filename, match=False, agb=False, track_data=None):
        '''
        filename [str] the path to the PMS or PMS.HB file
        '''
        (self.base, self.name) = os.path.split(filename)
        self.hb = False
        if 'hb' in self.name.lower():
            self.hb = True
        # will house error string(s)
        self.flag = None
        self.info = {}
        self.match = False
        if match:
            self.load_match_track(filename, track_data=track_data)
            self.track_mass()
            self.filename_info()
            self.match = True
        elif agb:
            self.load_agb_track(filename)
        else:
            self.load_track(filename)
            self.filename_info()
            if self.flag is None:
                self.track_mass()

        if self.flag is None:
            self.check_track()
            if not match:
                self.check_header_arg(loud=True)

    def check_track(self):
        '''check if age decreases'''
        try:
            age = np.round(self.data.AGE, 6)
        except AttributeError:
            age = np.round(self.data.logAge, 6)
        test = np.diff(age) >= 0
        if False in test:
            print('Track.__init__: track has age decreasing!!', self.mass)
            bads, = np.nonzero(np.diff(age) < 0)
            try:
                print('Parsec track: offensive MODEs:', self.data.MODE[bads])
            except AttributeError:
                from ..eep.critical_point import Eep
                eep = Eep()
                if self.hb:
                    nticks = eep.nticks_hb
                    names = eep.eep_list_hb
                else:
                    nticks = eep.nticks
                    names = eep.eep_list
                inds = [np.argmin(np.abs(np.cumsum(nticks)-b)) for b in bads]
                print('Match track: offensive inds:', bads)
                print('Near:', np.array(names)[inds])
                import pdb; pdb.set_trace()
        return

    def track_mass(self):
        ''' choose the mass based on the physical track starting points '''
        try:
            good_age, = np.nonzero(self.data.AGE > 0.2)
        except AttributeError:
            # match tracks have log age
            good_age = [[0]]
        if len(good_age) == 0:
            self.flag = 'unfinished track'
            self.mass = self.data.MASS[-1]
            return self.mass
        try:
            self.mass = self.data.MASS[good_age[0]]
        except:
            e = sys.exc_info()
            print('Problem with Mass in {0}, {1}'.format(self.name, e))
            self.mass = self.data.MASS[good_age[0]]

        ind = self.name.lower().split('.').index('pms')

        ext = self.name.split('.')[ind]

        fmass = float(self.name.split('_M')[1].split('.' + ext)[0])

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
        self.Mbol = z_sun - 2.5 * self.data.LOG_L
        return self.Mbol

    def calc_logg(self):
        '''
        cgs constant is -10.616 adds self.logg and returns logg
        '''
        self.logg = -10.616 + np.log10(self.mass) + 4.0 * self.data.LOG_TE - \
            self.data.LOG_L
        return self.logg

    def calc_core_mu(self):
        '''
        Uses X, Y, C, and O.
        '''
        xi = np.array(['XCEN', 'YCEN', 'XC_cen', 'XO_cen'])
        ai = np.array([1., 4., 12., 16.])
        # fully ionized
        qi = ai/2.
        try:
            self.muc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                     for i in range(len(xi))))
        except:
            xi = np.array(['H_CEN', 'HE_CEN', 'C_cen', 'O_cen'])
            self.muc = 1. / (np.sum((self.data[xi[i]] / ai[i]) * (1 + qi[i])
                                     for i in range(len(xi))))
        return self.muc

    def calc_lifetimes(self):
        self.tau_he = np.sum(self.data.Dtime[self.data.LY>0])
        try:
            coreh, = np.nonzero((self.data.LX > 0) & (self.data['XCEN'] > 0))
        except:
            coreh, = np.nonzero((self.data.LX > 0) & (self.data['H_CEN'] > 0))

        self.tau_h = np.sum(self.data.Dtime[coreh])
        return

    def filename_info(self):
        '''
        # get Z, Y into attrs: 'Z0.0002Y0.4OUTA1.74M2.30'
        '''
        Z, Ymore = self.name.split('Z')[1].split('Y')
        Y = ''
        for y in Ymore:
            if y == '.' or y.isdigit():
                Y += y
            else:
                break
        self.Z = float(Z)
        self.Y = float(Y)
        if hasattr(self, 'header'):
            try:
                self.ALFOV, = np.unique([float(l.replace('ALFOV', '').strip())
                                         for l in self.header if ' ALFOV ' in l])
            except:
                pass

        if hasattr(self.data, 'QHEL'):
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
        names = 'logAge', 'MASS', 'LOG_TE', 'LOG_L', 'logg', 'CO'
        '''
        def mbol2logl(m):
            try:
                logl = (4.77 - float(m)) / 2.5
            except TypeError:
                logl = (4.77 - m) / 2.5
            return logl

        self.col_keys = ['logAge', 'MASS', 'LOG_TE', 'LOG_L', 'logg', 'CO']

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

        self.data = data.view(np.recarray)
        return data

    def load_track(self, filename):
        '''
        reads PMS file into a record array. Stores header and footer as
        list of strings in self.header

        if no 'BEGIN TRACK' in file will add message to self.flags

        if no footers are found will add message to self.info
        '''

        with open(filename, 'r') as infile:
            lines = infile.readlines()

        # find the header
        begin_track = -1
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
            self.data = np.array([])
            self.col_keys = None
            self.flag = 'load_track error: no begin track'
            self.mass = float(self.name.split('_M')[1].replace('.PMS', '').replace('.HB', ''))
            return
        
        if len(lines) - begin_track <= 2:
            self.data = np.array([])
            self.col_keys = None
            self.flag = 'load_track error: no data after begin track'
            self.mass = float(self.name.split('_M')[1].replace('.PMS', '').replace('.HB', ''))
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

        col_keys = lines[begin_track].replace('#', '').strip().split()
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
                print('Problem with line {0} of {1}, {2}'.format(i, filename, e))
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
            self.calc_lifetimes()
        fmt = '%i %.3f %5.3f %.2f %.3f %.4g %.4g\n'
        efmt = '# %.3f %s: %s \n'
        if self.flag is not None:
            line += efmt % (self.Z, self.name, self.flag)
        elif self.hb:
            line += fmt % (0, self.Z, self.mass, self.ALFOV,
                           self.zahb_mcore, self.tau_he, 0.)
        else:
            line += fmt % (1, self.Z, self.mass, self.ALFOV,
                           self.final_mcore, self.tau_he, self.tau_h)
        return line

    def load_agb_track(self, filename, cut=True):
        '''
        Read Paola's tracks.
        Cutting out a lot of information that is not needed for MATCH
        col_keys = ['MODE', 'status', 'NTP', 'AGE', 'MASS', 'LOG_Mdot',
                    'LOG_L', 'LOG_TE', 'Mcore', 'Y', 'Z', 'PHI_TP', 'CO']
        usecols = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        '''
        def find_thermal_pulses(ntp):
            '''find the thermal pulsations'''
            uniq_tps, uniq_inds = np.unique(ntp, return_index=True)
            tps = np.array([np.nonzero(ntp == u)[0] for u in uniq_tps])
            return tps

        def find_quiessent_points(tps, phi):
            '''
            find the quiessent points
            The quiescent phase is the the max phase in each TP,
            i.e., closest to 1
            '''
            if tps.size == 1:
                qpts = np.argmax(phi)
            else:
                qpts = np.unique([tp[np.argmax(phi[tp])] for tp in tps])
            return qpts

        col_keys = ['MODE', 'status', 'NTP', 'AGE', 'MASS', 'LOG_Mdot',
                    'LOG_L', 'LOG_TE', 'Mcore', 'Y', 'Z', 'PHI_TP', 'CO']
        usecols = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]

        f = open(filename)
        line1 = f.readline()
        line2 = f.readline()

        if 'information' in line2:
            line1 = line1.replace('L_*', 'LOG_L')
            line1 = line1.replace('step', 'MODE')
            line1 = line1.replace('T_*', 'LOG_TE')
            line1 = line1.replace('M_*', 'MASS')
            line1 = line1.replace('age/yr', 'AGE')
            line1 = line1.replace('dM/dt', 'LOG_Mdot')
            line1 = line1.replace('M_c', 'Mcore')
            col_keys = line1.strip().replace('#', '').replace('lg', '').split()
            col_keys = self.add_to_col_keys(col_keys, line2)
            usecols = list(np.arange(len(col_keys)))

        data = np.genfromtxt(filename, names=col_keys, usecols=usecols)
        self.data = data.view(np.recarray)

        if self.data.NTP.size == 1:
            self.flag = 'no abg tracks'
            return
        self.Z = self.data.Z[0]
        self.Y = self.data.Y[0]
        self.mass = self.data.MASS[0]
        self.col_keys = col_keys

        # The first line in the agb track is 1. This isn't a quiescent stage.
        self.data.PHI_TP[0] = -99.

        self.tps = find_thermal_pulses(self.data.NTP)
        self.qpts = find_quiessent_points(self.tps, self.data.PHI_TP)
        return

    def add_to_col_keys(self, col_keys, additional_col_line):
        '''
        If fromHR2mags was run, A new line "Additional information Added:..."
        is added, this adds these column keys to the list.
        '''
        new_cols = additional_col_line.split(':')[1].strip().split()
        col_keys = list(np.concatenate((col_keys, new_cols)))
        return col_keys

    def maxmin(self, col, inds=None):
        '''
        return the max and min of a column in self.data, inds to slice.
        '''
        arr = self.data[col]
        if inds is not None:
            arr = arr[inds]
        ma = np.max(arr)
        mi = np.min(arr)
        return (ma, mi)

    def add_header_args_dict(self):
        def parse_args(header_line):
            header_line = header_line.replace('*', '')
            k, v = zip(*[a.split('=') for a in [l for l in header_line.split()
                                                if '=' in l and 'RESTART' not in l]])
            arg_dict = dict(zip(k, map(float, v)))
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
