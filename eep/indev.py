class InDevelopment(object):
    """
    half baked or unfinished
    """
    def first_check():
        ''' wip: toward eleminating the use of Sandro's eeps '''
        for i in range(70):
            ind = [farther(inds1, i, 5)]

    def farther(arr, ind, dist):
        ''' wip: toward eleminating the use of Sandro's eeps '''
        return arr[np.nonzero(np.abs(arr - arr[ind]) > dist)[0]]

    def recursive_farther(arr, dist, n):
        ''' wip: toward eleminating the use of Sandro's eeps '''
        saved = [arr[0]]
        for i in range(n):
            if i == 0:
                narr = farther(arr, 0, dist)
                print(i, narr)
            else:
                narr = farther(narr, 0, dist)
                print(i, narr)
            saved.append(narr[0])
            # if saved[i] == saved[i-1]:
            #    break
        return saved

    def convective_core_test(self, track):
        '''
        only uses sandro's defs, so doesn't need load_critical_points
        initialized.
        '''
        ycols = ['QSCHW', 'QH1', 'QH2']
        age_ = track.data[age]
        lage = np.log10(age_)

        morigs = [t for t in self.ptcri.data_dict['M%.3f' % self.mass]
                  if t > 0 and t < len(track.data[logL])]
        iorigs = [np.nonzero(track.data[MODE] == m)[0][0] for m in morigs]
        try:
            inds = self.ptcri.inds_between_ptcris(track, 'MS_BEG', 'POINT_B',
                                                  sandro=True)
        except IndexError:
            inds, = np.nonzero(age > 0.2)

        try:
            lage[inds]
        except IndexError:
            inds = np.arange(len(lage))
        plt.figure()
        ax = plt.axes()
        for ycol in ycols:
            ax.plot(lage[inds], track.data[ycol][inds], lw=2, label=ycol)
            ax.scatter(lage[iorigs[4]], track.data[ycol][iorigs[4]], s=100,
                       marker='o', color='red')
            if len(iorigs) >= 7:
                ax.scatter(lage[iorigs[7]], track.data[ycol][iorigs[7]], s=100,
                           marker='o', color='blue')
                xmax = lage[iorigs[7]] + .4
            else:
                xmax = ax.get_xlim()[1]

        xmin = lage[iorigs[4]] - .4
        ax.set_xlim(xmin, xmax)
        ax.set_title(self.mass)
        ax.legend(loc=0)

    def strip_instablities(self, track, inds):
        return track
        peak_dict = utils.find_peaks(track.data[logL][inds])
        extrema = np.sort(np.concatenate([peak_dict['maxima_locations'],
                                          peak_dict['minima_locations']]))
        # divide into jumps that are at least 50 models apart
        jumps, = np.nonzero(np.diff(extrema) > 50)
        if len(jumps) == 0:
            print('no istabilities found')
            return track
        if not hasattr(track, 'data_orig'):
            track.data_orig = copy.copy(track.data)
        # add the final point
        jumps = np.append(jumps, len(extrema) - 1)
        # instability is defined by having more than 20 extrema
        jumps = jumps[np.diff(np.append(jumps, extrema[-1])) > 20]
        # np.diff is off by one in the way I am about to use it
        # burn back some model points to blend better with the old curve...
        starts = jumps[:-1] + 1
        ends = jumps[1:]
        # the inds to smooth.
        for i in range(len(jumps)-1):
            moffset = (inds[extrema[ends[i]]] - inds[extrema[starts[i]]]) / 10
            poffset = moffset
            if i == len(jumps)-2:
                poffset = 0
            finds = np.arange(inds[extrema[starts[i]]] - moffset,
                              inds[extrema[ends[i]]] + poffset)
            tckp, step_size, non_dupes = self._interpolate(track, finds, s=0.2,
                                                           linear=True)
            arb_arr = np.arange(0, 1, step_size)
            if len(arb_arr) > len(finds):
                arb_arr = np.linspace(0, 1, len(finds))
            else:
                print(len(finds), len(xnew))
            agenew, xnew, ynew = splev(arb_arr, tckp)
            # ax.plot(xnew, ynew)
            track.data[logL][finds] = ynew
            track.data[logT][finds] = xnew
            track.data[age][finds] = agenew
            print('logL, logT, age interpolated from inds %i:%i' %
                  (finds[0], finds[-1]))
            track.header.append(
                'logL, logT, age interpolated from MODE %i:%i \n' %
                (track.data[MODE][finds][0], track.data[MODE][finds][-1]))

            self.check_strip_instablities(track)
        return track

    def check_strip_instablities(self, track):
        fig, (axs) = plt.subplots(ncols=2, figsize=(16, 10))
        for ax, xcol in zip(axs, [age, logT]):
            for data, alpha in zip([track.data_orig, track.data], [0.3, 1]):
                ax.plot(data[xcol], data[logL], color='k', alpha=alpha)
                ax.plot(data[xcol], data[logL], ',', color='k')

            ax.set_xlabel('$%s$' % xcol.replace('_', r'\! '), fontsize=20)
            ax.set_ylabel('$LOG\! L$', fontsize=20)
        plt.show()
        '''
        I'm currently not doing shit with this... it's such a fast track it
        wont make a big difference in MATCH.
        In any case, hopefully MATCH treats these reasons as blurs on a cmd
        as they should probably be treated...

        # there are instabilities in massive tracks that are on the verge or
        # returning to the hot side (Teff>10,000) of the HRD before TPAGB.
        # The following is designed to cut the tracks before the instability.
        # If the star is M>55. and the last model doesn't reach Teff = 10**4,
        # The track is cut at the max logL after the MS_TO, otherwise, that
        # value is the TPAGB (not actually TP-AGB, but end of the track).
        fin = len(track.data[logL]) - 1
        inds = np.arange(ms_to, fin)
        peak_dict = utils.find_peaks(track.data[logL][inds])
        max_frac = peak_dict['maxima_number'] / float(len(inds))
        min_frac = peak_dict['minima_number'] / float(len(inds))
        if min_frac > 0.15 or max_frac > 0.15:
            print('M=%.3f' % track.mass)
            print('MS_TO+ %i inds. Fracs: max %.2f, min %.2f' %
                  (len(inds), max_frac, min_frac))
            print('MS_TO+ minima_number', peak_dict['minima_number'])
            print('MS_TO+ maxima_number', peak_dict['maxima_number'])
            track = self.strip_instablities(track, inds)
        '''

    def check_sandros_eeps(self, track):
        '''
        Sometimes Sandro's points are wrong or poorly placed. This is a
        hack until I completely remove dependance on his eeps
        The reason the hack can work is because these eeps have no physical
        meaning, so adjusting them one or two points adjacent on the track
        should be fine.
        '''
        loud = True
        errmsg = 'Track is unworkable. Check Sandro\'s eeps'
        msg = '%s too close, moved this one away'
        eep = Eep()
        inds, = np.nonzero(track.iptcri > 0)
        idefined = np.array(track.iptcri)[inds]
        defined = np.array(eep.eep_list)[inds]

        if np.min(np.diff(idefined)) <= 1:
            same = np.argmin(np.diff(idefined))
            isame = idefined[same]

            after = same + 1
            iafter = idefined[after]

            before = same - 1
            ibefore = idefined[before]
            if iafter - ibefore < 5:
                # can't space them with one ind in between!
                # must move the before or after eep.
                earlier = eep.eep_list[eep.eep_list.index(defined[before]) - 1]
                later = eep.eep_list[eep.eep_list.index(defined[after]) + 1]
                iearlier = \
                    track.iptcri[self.ptcri.get_ptcri_name(earlier,
                                                           sandro=False)]
                ilater = track.iptcri[self.ptcri.get_ptcri_name(later,
                                                                sandro=False)]
                # is there space to adjust now?
                if ilater - iearlier < 9:
                    track.flag = errmsg
                    return

                # move the point toward where there is more room
                if iafter - isame < isame - ibefore:
                    inew = iafter + 1
                    inew_name = defined[after]
                else:
                    inew = ibefore - 1
                    inew_name = defined[before]
            else:
                # can shift the offender!
                if iafter - isame < isame - ibefore:
                    inew = isame - 1
                else:
                    inew = isame + 1
                inew_name = defined[same]

            self.add_eep(track, inew_name, inew, loud=loud,
                         message=msg % defined[same])

            if np.min(np.diff(track.iptcri)) <= 1:
                self.check_sandros_eeps(track)
