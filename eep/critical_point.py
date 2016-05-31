from __future__ import print_function
import os
import numpy as np
import sys

from ..config import *

import logging
from ..config import *
logger = logging.getLogger()

class Eep(object):
    '''
    a simple class to hold eep data. Gets added as an attribute to
    critical_point class.
    The lengths are then used in match.py
    '''
    def __init__(self):
        '''hard coded default eep_list and lengths
        eeps = {'PMS_BEG': 60,
                    'PMS_MIN': 60,
                    'PMS_END': 80,
                    'MS_BEG': 199,
                    'MS_TMIN': 100,
                    'MS_TO': 100,
                    'SG_MAXL': 70,
                    'RG_MINL': 370,
                    'RG_BMP1': 30,
                    'RG_BMP2': 400,
                    'RG_TIP': 40,
                    'HE_BEG': 150,
                    'YCEN_0.550': 100,
                    'YCEN_0.500': 60,
                    'YCEN_0.400': 100,
                    'YCEN_0.200': 80,
                    'YCEN_0.100': 80,
                    'YCEN_0.005': 80,
                    'YCEN_0.000': 40,
                    'AGB_LY1': 40,
                    'AGB_LY2': 20,
                    'TPAGB': -1}
        eep_list = eeps.keys()
        eep_lengths = eeps.values()
        '''
        eep_list = ['PMS_BEG', 'PMS_MIN', 'PMS_END', 'MS_BEG', 'MS_TMIN',
                    'MS_TO', 'SG_MAXL', 'RG_MINL', 'RG_BMP1', 'RG_BMP2',
                    'RG_TIP', 'HE_BEG', 'YCEN_0.550', 'YCEN_0.500',
                    'YCEN_0.400', 'YCEN_0.200', 'YCEN_0.100', 'YCEN_0.005',
                    'END_CHEB', 'TPAGB']
        eep_lengths = [80, 80, 80, 199, 100, 100, 70, 370, 30, 400, 40, 150,
                       100, 60, 100, 80, 80, 80, 80]
        assert (len(eep_list) == len(eep_lengths) + 1), 'Bad eep definitions'
        ihb = eep_list.index('HE_BEG')
        eep_list_hb = np.copy(eep_list[ihb:])
        eep_lengths_hb = np.copy(eep_lengths[ihb:])

        self.eep_list = eep_list
        self.nticks = eep_lengths
        self.eep_list_hb = eep_list_hb
        self.nticks_hb = eep_lengths_hb

        # usefull to check match compatibility
        ims = eep_list.index('MS_TO')
        trans = ihb - 1
        self.nlow = np.sum(eep_lengths[:ims])
        self.nhb = np.sum(eep_lengths_hb)
        self.nms = np.sum(eep_lengths[:trans])
        self.ntot = self.nms + self.nhb + eep_lengths[trans]
        self.trans = eep_lengths[trans]

class critical_point(object):
    '''
    class to hold ptcri data from Sandro's ptcri file and input eep_obj
    which tells which critical points of Sandro's to ignore and which new
    ones to define. Definitions of new eeps are in the Track class.
    '''
    def __init__(self, filename=None, sandro=True, debug=False, hb=False):
        self.debug = debug

        if filename is not None:
            if 'hb' in filename:
                hb = True
            self.base, self.name = os.path.split(filename)
            self.load_ptcri(filename, sandro=sandro, hb=hb)
            self.get_args_from_name(filename)
        else:
            load_eep(hb=hb)

    def load_eep(self, hb=False):
        self.eep = Eep()
        if hb:
            self.eep_list = self.eep.eep_list_hb
        else:
            self.eep_list = self.eep.eep_list

        self.key_dict = dict(zip(self.eep_list, range(len(self.eep_list))))
        self.please_define = self.eep_list

    def get_args_from_name(self, filename):
        '''
        strip Z and Y and add to self must have format of
        ..._Z0.01_Y0.25...
        '''
        zstr = filename.split('_Z')[-1]
        try:
            self.Z = float(zstr.split('_')[0])
        except:
            self.Z = float(zstr.split('Y')[0])
        ystr = filename.replace('.dat', '').split('_Y')[-1].split('_')[0]
        if ystr.endswith('.'):
            ystr = ystr[:-1]
        try:
            self.Y = float(ystr)
        except:
            ystr = filename.replace('.dat', '').split('Y')[-1].split('_')[0]
            if ystr.endswith('.'):
                ystr = ystr[:-1]
            self.Y = float(ystr)

    def inds_between_ptcris(self, track, name1, name2, sandro=True):
        '''
        returns the indices from [name1, name2)
        this is iptcri, not mptcri (which start at 1 not 0)
        they will be the same inds that can be used in Track.data
        '''
        def getind(inds, name, sandro=sandro):
            try:
                ind = inds[self.get_ptcri_name(name, sandro=sandro)]
            except IndexError:
                ind = 0
            return ind
        
        if sandro:
            # this must be added in Tracks.load_critical_points!
            inds = track.sptcri
        else:
            inds = track.iptcri

        first = getind(inds, name1, sandro=sandro)
        second = getind(inds, name2, sandro=sandro)

        return np.arange(first, second)


    def get_ptcri_name(self, val, sandro=True, hb=False):
        '''
        given the eep number or the eep name return the eep name or eep number.
        '''
        if sandro:
            pdict = self.sandros_dict
        else:
            pdict = self.key_dict

        if type(val) == int:
            return [name for name, pval in pdict.items() if pval == val][0]
        elif type(val) == str:
            return [pval for name, pval in pdict.items() if name == val][0]

    def load_ptcri(self, filename, sandro=True, hb=False):
        '''
        Read the ptcri*dat file.
        Initialize Eep
        Flag the missing eeps in the ptcri file.
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()

        # the lines have the path name, and the path has F7.
        if not hb:
            begin, = [i for i in range(len(lines)) if lines[i].startswith('#')
                      and 'F7' in lines[i]]
        else:
            begin = -1
            if 'p2m' in filename:
                begin = 0

        if sandro and not hb:
            try:
                self.fnames = [l.strip().split('../F7/')[1]
                               for l in lines[(begin+2):]]
            except IndexError:
                # last two lines of Sandro's files have a different format
                self.fnames = [l.strip().split('../F7/')[1]
                               for l in lines[(begin+2):-2]]

        # the final column is a filename.
        all_keys = lines[begin + 1].replace('#', '').strip().split()
        col_keys = all_keys[3:-1]
        # ptcri file has filename as col #19 so skip the last column
        usecols = range(0, len(all_keys) - 1)
        if hb and not 'p2m' in filename:
            col_keys = all_keys[3:]
            usecols = range(0, len(all_keys))
        try:
            col_keys[col_keys.index('C_BUR')] = 'TPAGB'
        except ValueError:
            pass
        try:
            col_keys[col_keys.index('HeLST')] = 'TPAGB'
        except ValueError:
            pass
        # invalid_raise will skip the last rows that Sandro uses to fake the
        # youngest MS ages (600Msun).
        data = np.genfromtxt(filename, usecols=usecols, skip_header=begin + 2,
                             invalid_raise=False)
        self.data = data
        self.masses = data[:, 1]

        data_dict = {}
        for i in range(len(data)):
            str_mass = 'M%.3f' % self.masses[i]
            data_dict[str_mass] = data[i][3:].astype(int)

        self.data_dict = data_dict

        self.load_eep(hb=hb)

        if sandro:
            # loading sandro's eeps means they will be used for match
            self.sandro_eeps = col_keys
            self.sandros_dict = dict(zip(col_keys, range(len(col_keys))))
            self.please_define = [c for c in self.eep_list if c not in col_keys]

            # there is no mixture between Sandro's HB eeps since there
            # are no HB eeps in the ptcri files. Define them all here.
            #self.please_define_hb = eep_obj.eep_list_hb

            [self.check_ptcri(self.masses[i], data[i][3:].astype(int))
             for i in range(len(data))]

    def load_eeps(self, track, sandro=True):
        '''load the eeps from the ptcri file'''
        try:
            ptcri = self.data_dict['M%.3f' % track.mass]
        except KeyError:
            track.flag = 'No M%.3f in ptcri.data_dict.' % track.mass
            return track
        
        if sandro:
            track.sptcri = \
                np.concatenate([np.nonzero(track.data[MODE] == m)[0]
                                for m in ptcri])
        else:
            track.iptcri = ptcri
        track.ptcri_file = os.path.join(self.base, self.name)
        return track

    def save_ptcri(self, tracks, filename=None, hb=False):
        '''save parsec2match ptcris in same format as sandro's'''

        if filename is None:
            filename = os.path.join(self.base, 'p2m_%s' % self.name)
            if hb:
                filename = filename.replace('p2m', 'p2m_hb')

        sorted_keys, inds = zip(*sorted(self.key_dict.items(),
                                        key=lambda (k, v): (v, k)))

        header = '# critical points in F7 files defined by sandro, basti, and phil \n'
        header += '# i mass kind_track %s fname \n' % (' '.join(sorted_keys))
        with open(filename, 'w') as f:
            f.write(header)
            linefmt = '%2i %.3f 0.0 %s %s \n'
            for i, track in enumerate(tracks):
                if track.flag is not None:
                    print('save_ptcri skipping %s: %s' % (track.name, track.flag))
                    continue
                ptcri_str = ' '.join(['%5d' % p for p in track.iptcri])
                f.write(linefmt % (i+1, track.mass, ptcri_str,
                                   os.path.join(track.base, track.name)))
        #print('wrote %s' % filename)

    def check_ptcri(self, mass_, arr):
        ndefined = len(np.nonzero(arr > 0)[0])
        needed = 15

        if (mass_ > inte_mass) and (mass_ <= high_mass):
            if ndefined != needed:
                print('check_ptcri error: M%.3f does not have enough EEPs' % mass_)
                try:
                    masses = np.array([f.split('F7_M')[1].replace('.PMS', '')
                                       for f in self.fnames], dtype=float)
                except:
                    masses = np.array([f.split('F7_M')[1].replace('.DAT', '')
                                       for f in self.fnames], dtype=float)

                inds, = np.nonzero(mass_ == masses)
                print('files in question:')
                print(np.array(self.fnames)[inds])
                for ind in inds:
                    self.fix_ptcri(np.array(self.fnames)[ind])

    def fix_ptcri(self, fname, iptcri=None):
        """
        print better values of sandro's EEPs (the basis for Phil's EEPs)
        (It is expected you will hand code the proper values in the ptcri file
        or a copy of the ptcri file.)

        This was made for inte_mass tracks, 12>M>20 that had errors.
        The errors were typically:
        1) MS_BEG is at the point in the track that should be POINT_C
        2) POINT_C is at the point in the track that should be LOOP_B
        3) RG_TIP is at the point in the track that should probably be the final
            EEP.
        4) LOOP_A, B, C, TPAGB/CBUR EEPS are all 0.
            This final issue could be accounted for in the ptcri reader, but
            it is taken as a sign that the other EEPs are probably wrong.)

        Paramaters
        ----------
        fnane : string
            filename of the F7 file from the bad line of the ptcri file

        Returns
        -------
            sys.exit()
            prints EEPs to stout and exits upon completion.
        """
        import matplotlib.pylab as plt
        plt.ion()
        from ..fileio import get_dirs, get_files
        from ..tracks import TrackDiag, Track
        td = TrackDiag()

        def guessandcheck(ptname, pt=None):
            """
            interactively plot an input point on a track's HRD.
            Enter the index of the array to plot, and 0 to return that index.

            A big assumption is that the ptcri file is in a folder called data
            and that the corresponding track is in ../tracks/
            Parameters
            ----------
            ptname : string
                name of the track point, for stout messages only

            pt : int optional
                a starting guess. If not None, will plot it first.

            Returns
            -------
            pt : int
                the final value decided
            """
            go_on = 1

            outmsg = '%s MODE: %i'
            if pt is not None:
                # plot guess first
                print(outmsg % (ptname, track.data[MODE][pt]))
                go_on, pt = plot_point(ptname, pt)

            while go_on != 0:
                go_on, pt = plot_point(ptname, pt)

            print(outmsg % (ptname, track.data[MODE][pt]))
            return track.data[MODE][pt]

        def plot_point(ptname, pt):
            inmsg = 'Enter new value for %s or 0 to move on: ' % ptname
            ax.plot(track.data[logT][pt], track.data[logL][pt], '*')
            plt.draw()
            go_on = raw_input(inmsg)
            try:
                go_on = int(go_on)
            except:
                import pdb; pdb.set_trace()
            if go_on != 0:
                # don't overwrite with "move on" value
                pt = go_on
            return go_on, pt

        #track_dir = self.base.replace('data', 'tracks')
        track_dir, tname = os.path.split(fname)
        estr = 'PMS'
        hb = 'hb' in fname.lower()
        if hb:
            estr = 'HB'
        z = tname.split('Z')[1].split('Y')[0].replace('_', '')
        mass_ = tname.split('M')[1].replace('.DAT', '').replace('.P', '').split('.HB')[0]
        #track_dir, = get_dirs(track_dir, '{:g}'.format(z))

        #track_file = os.path.join(track_dir, '/'.join(fname.split('/')[1:]))
        try:
            track_file, = get_files(track_dir, '*M{}*{}'.format(mass_, estr))
        except ValueError:
            # the above search text will not distiguish, e.g, M030.000 and M130.000
            track_files = get_files(track_dir, '*{}*'.format(mass_))
            # cull mass values
            import pdb; pdb.set_trace()
            tms = np.array(['.'.join(os.path.split(t)[1].split('M')[1].split('.')[:-1])
                            for t in track_files], dtype=float)
            # select by mass
            track_file, = np.array(track_files)[np.nonzero(float(mass_) == tms)]


        track = self.load_eeps(Track(track_file))

        if iptcri is not None:
            track.iptcri = iptcri
            fig, ax = plt.subplots()
            ax.plot(track.data[logT], track.data[logL])
            ax.plot(track.data[logT][iptcri], track.data[logL][iptcri], 'o')
            td.annotate_plot(track, ax, logT, logL)
        else:
            ax = td.plot_sandro_ptcri(track, ptcri=self)

        print('Open sandros ptcri file and get ready to edit. {}'.format(tname))
        print('Current values:', track.sptcri)
        if len(track.sptcri) <= 3:
            print('this ptcri looks especially messed up.')
            import pdb; pdb.set_trace()

        pms_beg = guessandcheck('pms_beg', pt=track.sptcri[0])
        pms_min = guessandcheck('pms_min', pt=track.sptcri[1])
        pms_end = guessandcheck('pms_end', pt=track.sptcri[2])
        near_zam = guessandcheck('near_zam', pt=track.sptcri[4])
        ms_beg = guessandcheck('ms_beg', pt=track.sptcri[4])

        # experience has shown that Sandro's code sets MS_BEG as what should
        # be point C
        point_c = track.sptcri[4]
        point_c = guessandcheck('point_c', pt=point_c)

        # MS_TMIN is the min LOG_TE between MS_BEG and POINT_C
        inds = np.arange(ms_beg, point_c, dtype=int)
        point_b = ms_beg + np.argmin(track.data[logT][inds])
        point_b = guessandcheck('point_b', pt=point_b)

        # RG_BASE is probably the lowest LOG_L after POINT_C
        inds = np.arange(point_c, len(track.data), dtype=int)
        rg_base = point_c + np.argmin(track.data[logL][inds])
        rg_base = guessandcheck('rg_base', pt=rg_base)

        # RG_TIP is the peak LOG_L after RG_BASE, probably happens soon
        # in high mass tracks.
        inds = np.arange(rg_base, rg_base + 200, dtype=int)
        rg_tip = rg_base + np.argmax(track.data[logL][inds])
        rg_tip = guessandcheck('rg_tip', pt=rg_tip)

        # There is no RG_BMP when there is core fusion. Pick equally spaced
        # points
        rgbmp1, rgbmp2 = np.linspace(rg_base, rg_tip, 4, endpoint=False)[1:-1]

        # Take the final point of the track, either the final, or where
        # Sandro cut it
        fin = track.data[MODE][-1]
        fin_sandro = track.sptcri[np.nonzero(track.sptcri>0)][-1]
        if fin > fin_sandro:
            print('last point in track %i, Sandro cut at %i.' %  (fin, fin_sandro))
            fin = guessandcheck('final point', pt=fin_sandro)

        # I don't use Sandro's loops, so I don't care, here are non zero points that
        # ensure age increase.
        loopa, loopb, loopc = np.linspace(rg_tip, fin, 5, endpoint=False)[1:-1]


        print(track.sptcri[:3])
        print('suggested line ptcri line:')
        print(''.join(['%10i' % i for i in (pms_beg, pms_min, pms_end, near_zam,
                                            ms_beg, point_b, point_c, rg_base,
                                            int(rgbmp1), int(rgbmp2), rg_tip,
                                            loopa, loopb, loopc, fin)]))

        sys.exit()

if __name__ == "__main__":
    ptcri = critical_point(filename=sys.argv[1])
