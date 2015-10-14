""" NOT USED CURRENTLY"""
"""
class f4_file(object):
    def __init__(self, filename):
        self.base, self.name = os.path.split(filename)
        self.load_f4_file(filename)

    def load_f4_file(self, filename):
        '''
        #S MODELL              ALTER        Q_BOT        QINTE    TINTE         B_SLX        B_SLNU
        #H MODELL                SLX        T_BOT         QHEL     THEL         B_SLY         B_SEG
        #C    CNO                SLY       RH_BOT          lgL     lgTe         B_SLC        HM_CHE
        Rg MODELL                 V1           V2           V3       V4            V5            V6        H            HE3          HE4          C            C13          N14          N15          O16          O17          O18          NE20         NE22         MG25         LI7          BE7          F19          MG24         MG26         NE21         NA23         AL26         AL27         SI28         Deut         ZH
        S       1  0.10000000000E+00  1.000000000  0.000000000   7.5711  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        H       0  0.10239192793E+01  0.000000000  0.000000000   0.0000  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        C      -2  0.00000000000E+00  0.000000000  6.112131886   4.6639  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        S       2  0.10000000000E+00  1.000000000  0.000000000   7.5705  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        H       0  0.10018991298E+01  0.000000000  0.000000000   0.0000  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
        C      -2  0.00000000000E+00  0.000000000  6.111904543   4.6636  0.000000E+00  0.000000E+00 7.000000E-01 4.400000E-05 2.800000E-01 3.610000E-03 4.440000E-05 1.200000E-03 4.740000E-06 1.080000E-02 4.190000E-06 2.440000E-05 5.730000E-04 7.700000E-05 1.840000E-04 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00


        F4 file is in blocks of 3 rows for each model
        Row 1:  SURFACE  (S)
        Row 2:  HBURNING (H)
        Row 3:  CENTRE   (C)

        For each block the columns are:
        1: the region (S,H,C)

        2S:     MODELL  = the model number
        2H:     allways = 0 not defined
        2C:     CNO burning still not defined

        3S:   ALTER  age
        3H:     SLX  LX/L_tot_surf
        3C:     SLY  LY/L_tot_surf

        4S:     Q_BOT     m/Mtot  at the bottom of the envelope convective region (including envelope overshoot)
        4H:     T_BOT     log10(T)
        4C:    RH_BOT     log10(rho)

        5S:    QINTE     m/Mtot where H=0.5*Hsurf as in F7
        5H:     QHEL     max(m/Mtot where H=0.)   as in F7
        5C:      lgL     Surface total luminosity as in F7  (L_TOT)

        6S:   TINTE      log10(T) at QINTE
        6H:    THEL      log10(T) at QHEL
        6C:    lgTe      surface Te (log10)

        7S:   B_SLX      L_H/L_tot_surf at the bottom of the conv envelope
        7H:   B_SLY      same for He
        7C:   B_SLC      same for Carbon

        7S:   B_SLNU     same for neutrinos
        7H:    B_SEG     same for gravitational energy (L_GRAV/L_TOT)
        7C:   HM_CHE     min time step size in chemistry routine

        8-end_S:  composition as indicated, H HE3 etc.. at the surface
        8-end_H:  composition as indicated, H HE3 etc.. at the H zone
        8-end_C:  composition as indicated, H HE3 etc.. at the C zone
        '''
        #'/Users/phil/research/BRratio/models/model_grid/PH_COV0.5_ENV0.50_Z0.01_Y0.2663/PH_COV0.5_ENV0.50_Z0.01_Y0.2663/Z0.01Y0.2663OUTA1.74_F4_M5.00'
        #import copy
        #data = fileIO.readfile(filename, col_key_line=3)
        self.surface = fileIO.readfile(filename, col_key_line=3)[::3]
        self.hburning = fileIO.readfile(filename, col_key_line=3)[1::3]
        self.center = fileIO.readfile(filename, col_key_line=3)[2::3]
        self.surface.dtype.names = tuple('Surface MODE ALTER Q_BOT QINTE TINTE B_SLX B_SLNU'.split()) + self.surface.dtype.names[8:]
        self.hburning.dtype.names = tuple('Hburning MODE SLX T_BOT QHEL THEL B_SLY B_SEG'.split()) + self.hburning.dtype.names[8:]
        self.center.dtype.names = tuple('Center CNO SLY RH_BOT LOG_L LOG_TE B_SLC HM_CHE'.split()) + self.center.dtype.names[8:]
"""