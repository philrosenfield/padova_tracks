import numpy as np

class IsoTrack2(object):
    def __init__(self, filename):
        self.read_ptcri2(filename)
        self.base, self.name = os.path.split(filename)

    def read_dbert2(self, filename):
        '''
        These are *HB, *INT, *LOW files that leo processes into *HB2, etc
        header fmt:
        2 # number of tracks sets with same eeps.
        1 18 # which isochrones have first eep set
        19 26 # which isochrones have the second eep set
        26 # number of isochrones
        384 398 395 397 392 ... number of age steps in each isochrone
        1.750000 1.800000 1.850000 ... masses of each isochrone
        '''
        lines = [l.strip() for l in open(filename, 'r').readlines()]
        # file made to be read by C, going line by line for header info.
        eep_sets = int(lines[0])
        set_inds = np.array([lines[i + 1].split() for i in range(eep_sets)], dtype=int)
        set_inds[:, 0] -= 1
        nsets = int(lines[eep_sets + 1])
        len_tracks = np.array(lines[eep_sets + 2].split(), dtype=int)
        masses = np.array(lines[eep_sets + 3].split(), dtype=float)
        start_ind = eep_sets + 4

        start_inds = np.cumsum(len_tracks) + start_ind
        start_inds = np.insert(start_inds, 0, start_ind)
        end_inds = start_inds - 1

        # each one is too long
        end_inds = end_inds[1:]
        start_inds = start_inds[:-1]

        self.eep_sets = eep_sets
        self.nsets = nsets
        self.masses = masses
        for i in range(eep_sets):
            set_start, set_end = set_inds[i]
            isoc_strs = [lines[start_inds[j]: end_inds[j]] for j in np.arange(*set_inds[i])]
            isoc_data = np.array([np.array([isoc_strs[j][k].split()[:3] for k in range(len(isoc_strs[j]))], dtype=float) for j in range(len(isoc_strs))])
            isoc_masses = ['%.4f' % m for m in masses[np.arange(*set_inds[i])]]
            self.__setattr__('iso_str%i' % (i + 1), {isoc_masses[j]: isoc_strs[j] for j in range(len(isoc_masses))})
            self.__setattr__('iso_data%i' % (i + 1), {isoc_masses[j]: isoc_data[j] for j in range(len(isoc_masses))})
# get the EEPS too?? I donno what you need the data for besides plotting...

"""
self.data[logT]
self.data[age]
self.data.MODE
self.data[logL]
self.sptrci
XCEN
YCEN
LX
LY
"""