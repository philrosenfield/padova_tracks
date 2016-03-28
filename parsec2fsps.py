from ezpadova import cmd
import numpy as np


def redefine_phase(tab):
    # Evolutionary phases defined in the isochrones:
    # 0: main sequence
    # 1: sub giant branch
    # 2: red giant branch
    # 3: red horizontal branch / red clump (core He burning)
    # 4: early asymptotic giant branch
    # 5: thermally-pulsating AGB (composition sets C-rich vs. O-rich)
    # 6: Post-AGB
    # 7: blue stragglers
    # 8: blue horizontal branch
    # 9: Wolf-Rayet (composition sets WC vs. WN)
    # -1: transition from RGB tip to HB

    # Leo's -- Charlie's
    # 0 PMS -- 0
    # 1 MS -- 0
    # 2 SUBGIANT -- 1
    # 3 RGB -- 2
    # 4 HEB -- 3
    # 5 RHEB -- 3
    # 6 BHEB -- 3
    # 7 EAGB -- 4
    # 8 TPAGB -- 5
    # 9 POSTAGB -- 6
    # 10 WD -- 0
    
    tab['stage'][tab['stage'] == 1] = 0.
    tab['stage'][tab['stage'] == 2] = 1.
    tab['stage'][tab['stage'] == 3] = 2.
    tab['stage'][tab['stage'] == 4] = 3.
    tab['stage'][tab['stage'] == 5] = 3.
    tab['stage'][tab['stage'] == 6] = 3.
    tab['stage'][tab['stage'] == 7] = 4.
    # don't trust cmd2.8's tp-agb
    tab['stage'][tab['stage'] == 8] = 0.
    tab['stage'][tab['stage'] == 9] = 6.
    tab['stage'][tab['stage'] == 10] = 0.
    itp, = np.nonzero(tab['slope'])
    tab['stage'][itp] = 5.
    
    # you never know...
    tab['stage'][tab['stage'] > 10] = 0.
    return tab
    
def isoch_for_fsps():
    # these are the grid point Zs so cmd2.8 does not have to interpolate
    # (Z=0.05 might have been added)
    zs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.014,
          0.017,  0.02, 0.03, 0.04, 0.06]
    header = '# log(age) Mini Mact logl logt logg Composition Phase\n'
    fmt = '%.2f %.8f %.4f %.4f %.4f %.4f %.4f %.4f'
    
    keys = ['logageyr', 'M_ini', 'M_act', 'logL', 'logT', 'logg', 'CO', 'stage']
    agemin = 5.5
    agemax = 10.10  # cmd2.8 will not take lage >= 10.13
    for z in zs:
        outfile = 'isoc_z{:.4f}.dat'.format(z)
        tab = cmd.get_t_isochrones(agemin, agemax, 0.05, z, model='parsec12s_r14')
        tab = redefine_phase(tab)
        _, aidx = np.unique(tab['logageyr'], return_index=True)
        aidx = np.append(aidx, len(tab))
        idxes = [np.arange(aidx[i], aidx[i+1]) for i in range(len(aidx)-1)]
        with open(outfile, 'w') as outp:
            for idx in idxes:
                # stick in the header after each isochrone
                outp.write(header)
                np.savetxt(outp, tab[keys][idx], fmt=fmt)
        print('wrote {}'.format(outfile))
    return
