from __future__ import print_function
from ..tracks import TrackSet, Track
from .. import parsec2match
from ..fileio import InputParameters
import numpy as np
import os

def compare_track_sets():
    """work in progress ip"""
    indict1 = {'tracks_dir': '/Users/phil/research/stel_evo/CAF09_S12D_NS/tracks/',
               'set_name': 'CAF09_S12D_NS', 'prefixs': 'all', 'hb': False,
               'masses': '0.75 <= %f <= 12', 'do_interpolation': False}

    indict2 = {'tracks_dir': '/Users/phil/research/stel_evo/CAF09_V1.2S_M36/tracks/',
               'set_name': 'CAF09_V1.2S_M36', 'prefixs': 'all', 'hb': False,
               'masses': '0.75 <= %f <= 12', 'do_interpolation': False}

    indict2 = {'tracks_dir': '/Users/phil/research/stel_evo/CAF09_V1.2S_M36/tracks/',
               'set_name': 'CAF09_V1.2S_M36', 'prefixs': 'all', 'hb': False,
               'do_interpolation': False}
    np.unique(compts.track_sets[1].Zs), np.unique(compts.track_sets[0].Zs)

    [m for m in compts.track_sets[0].masses if not m in compts.track_sets[1].masses]
    [m for m in compts.track_sets[1].masses if not m in compts.track_sets[0].masses]
    [i for i,m in enumerate(compts.track_sets[0].masses) if m in compts.track_sets[1].masses]
    inds1 = [i for i,m in enumerate(compts.track_sets[0].masses) if m in compts.track_sets[1].masses]
    inds2 = [i for i,m in enumerate(compts.track_sets[1].masses) if m in compts.track_sets[0].masses]
    compts.track_sets[0].masses[inds1] == compts.track_sets[1].masses[inds2]


    '''print 'S12'
    print [[[' '.join(('%g' % t.Z, '%g' % t.mass, '%g' % float(h.strip().split()[1])))
            for h  in t.header if ' ALFOV ' in h] for t in ts.tracks] for ts in s12_ts]


    [[[(t.Z, '%g' % t.mass, '%g' % t.cov, '%g' % float(h.strip().split()[1]))
       for h  in t.header if ' ALFOV ' in h] for t in ts.tracks]
     for ts in cov_ts]
    '''

class CompareTrackSets(object):
    '''class to compare two sets of stellar evolution models'''
    def __init__(self, indict1, indict2):
        self.load_track_sets(indict1, indict2)

    def load_track_sets(self, indict1, indict2):
        '''load each track set'''
        default_dict = parsec2match.initialize_inputs()
        self.inputs = []
        self.set_names = []
        self.track_sets = []
        inp1 = InputParameters(default_dict=default_dict)
        inp1.add_params(indict1)
        prefixs1 = parsec2match.set_prefixs(inp1)
        inp2 = InputParameters(default_dict=default_dict)
        inp2.add_params(indict2)
        prefixs2 = parsec2match.set_prefixs(inp2)

        for i in range(len(prefixs2)):
            inp2.prefix = prefixs2[i]
            inp1.prefix = prefixs1[i]
            ts1 = TrackSet(inputs=inp1)
            ts1.load_characteristics()
            ts2 = TrackSet(inputs=inp2)
            ts2.load_characteristics()
            np.max([np.max(ts1.tracks[i].data.LOG_L - ts2.tracks[i].data.LOG_L) for i in range(len(ts1.tracks))])
                #track_set = np.append(track_set, ts)
        self.track_sets = np.append(self.track_sets, ts1)
        self.track_sets = np.append(self.track_sets, ts2)
        #self.set_names = np.append(self.set_names, inp.set_name)

        return


def diagnostic_table(dirname):
    """
    Walk through a directory that houses directories that have tracks and
    print their summaries to a file. See Track.summary for info.
    Assumes track names have "Z0." and their directories also have "Z0."
    Skips over any other directory.
    """
    with open('%s_track_summary.dat' % dirname, 'w') as outf:
        # header
        outf.write('# track Z mass ALFOV QHEL tau_He tau_H\n')
        for root, dirs, files in os.walk(dirname):
            if len(files) == 0:
                continue
            if not 'Z0.' in root:
                continue
            files = [f for f in files if f.endswith('PMS') or f.endswith('HB')]
            # Mix
            line = '# %s\n' % os.path.split(root)[1]
            print(line.strip())
            outf.write(line)
            # sort tracks by mass, makes HB tracks line up near PMS.
            tracks = sorted(files, key = lambda x: float(x.split('7_M')[1].replace('.PMS', '').replace('.HB', '')))
            for track in tracks:
                t = Track(os.path.join(root, track))
                line = t.summary()
                print(line.strip())
                outf.write(line)

if __name__ == '__main__':
    import sys
    diagnostic_table(sys.argv[1])
