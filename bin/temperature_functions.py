#!/usr/bin/env python3
import sys
import os
import logging
import numpy as np
import pandas as pd
import dateutil


def tempF2C(x): return (x-32.0)*5.0/9.0
def tempC2F(x): return (x*9.0/5.0)+32.0


def load_temperature_hdf5(temps_fn, local_time_offset, basedir=None, start_year=None, truncate_to_full_day=False):
    ## Load temperature
    # temps_fn = "{}_AT_cleaned.h5".format(station_callsign)
    logging.info("Using saved temperatures file '{}'".format(temps_fn))
    if basedir is not None:
        temps_fn = os.path.join(basedir, temps_fn)
    tempdf = pd.read_hdf(temps_fn, 'table')

    tmp = local_time_offset.split(':')
    tmp = int(tmp[0])*3600+int(tmp[1])*60
    sitetz = dateutil.tz.tzoffset(local_time_offset, tmp)
    tempdf.index = tempdf.index.tz_convert(sitetz)

    if truncate_to_full_day:
        x = tempdf.index[-1]
        if x.hour != 23:
            x = x-pd.Timedelta(days=1)
            tmp = '{:04d}-{:02d}-{:02d}'.format(x.year, x.month, x.day)
            tempdf = tempdf.loc[:tmp]
    if start_year is not None:
        tempdf = tempdf.loc['{}-01-01'.format(start_year):]
    logging.info("Temperature data date range used: {} through {}".format(tempdf.index[0], tempdf.index[-1]))
    return tempdf


def load_temperature_csv(fn, local_time_offset=None):
    t = pd.read_csv(fn, index_col=0)
    if local_time_offset is not None:
        tmp = local_time_offset.split(':')
        tmp = int(tmp[0])*3600+int(tmp[1])*60
        sitetz = dateutil.tz.tzoffset(local_time_offset, tmp)
        t.index = pd.to_datetime(t.index).tz_localize('UTC').tz_convert(sitetz)
    return t


# Function which computes BM (single sine method) degree day generation from temperature data
def compute_BMDD_Fs(tmin, tmax, base_temp, dd_gen):
    # Used internally
    def _compute_daily_BM_DD(mint, maxt, avet, base_temp):
        """Use standard Baskerville-Ermin (single sine) degree-day method
        to compute the degree-day values for each a single day.
        """
        if avet is None:
            avet = (mint+maxt)/2.0 # simple midpoint (like in the refs)
        dd = np.nan # value which we're computing
        # Step 1: Adjust for observation time; not relevant
        # Step 2: GDD = 0 if max < base (curve all below base)
        if maxt < base_temp:
            dd = 0
        # Step 3: Calc mean temp for day; already done previously
        # Step 4: min > base; then whole curve counts
        elif mint >= base_temp:
            dd = avet - base_temp
        # Step 5: else use curve minus part below base
        else:
            W = (maxt-mint)/2.0
            tmp = (base_temp-avet) / W
            if tmp < -1:
                print('WARNING: (base_temp-avet)/W = {} : should be [-1:1]'.format(tmp))
                tmp = -1
            if tmp > 1:
                print('WARNING: (base_temp-avet)/W = {} : should be [-1:1]'.format(tmp))
                tmp = 1
            A = np.arcsin(tmp)
            dd = ((W*np.cos(A))-((base_temp-avet)*((np.pi/2.0)-A)))/np.pi
        return dd

    # compute the degree-days for each day in the temperature input (from tmin and tmax vectors)
    dd = pd.concat([tmin,tmax], axis=1)
    dd.columns = ['tmin', 'tmax']
    dd['DD'] = dd.apply(lambda x: _compute_daily_BM_DD(x[0], x[1], (x[0]+x[1])/2.0, base_temp), axis=1)
    # compute the degree-days for each day in the temperature input (from a daily groupby)
#     grp = t.groupby(pd.TimeGrouper('D'))
#     dd = grp.agg(lambda x: _compute_daily_BM_DD(np.min(x), np.max(x), None, base_temp))
#     dd.columns = ['DD']

    # Find the point where cumulative sums of degree days cross the threshold
    cDD = dd['DD'].cumsum(skipna=True)
    for cumdd_threshold,label in [[1*dd_gen,'F1'], [2*dd_gen,'F2'], [3*dd_gen,'F3']]:
        dtmp = np.zeros(len(dd['DD']))*np.nan
        tmp = np.searchsorted(cDD, cDD+(cumdd_threshold)-dd['DD'], side='left').astype(float)
        tmp[tmp>=len(tmp)] = np.nan
        #dd[label+'_idx'] = tmp
        # convert those indexes into end times
        e = pd.Series(index=dd.index)#, dtype='datetime64[ns]')

        #e[~np.isnan(tmp)] = dd.index[tmp[~np.isnan(tmp)].astype(int)] # @TCC previous code
        e.loc[~np.isnan(tmp)] = dd.index[tmp[~np.isnan(tmp)].astype(int)]
        e.loc[np.isnan(tmp)] = np.nan
        dd[label+'_end'] = e
        # and duration...
        #dd[label] = (e-dd.index+pd.Timedelta(days=1)).apply(lambda x: np.nan if pd.isnull(x) else x.days) # @TCC previous code
        dd[label] = (pd.to_datetime(e)-dd.index+pd.Timedelta(days=1)).apply(lambda x: np.nan if pd.isnull(x) else x.days)
        #dd.loc[np.isnan(tmp), label] = np.nan

    print("DD dataframe min values\n", dd.min())
    return dd


def compute_year_over_year_norm(in_dataframe,
                                start, end,
                                norm_start=None, norm_end=None,
                                freq='daily',
                                interp_method='linear',
                                norm_method='mean'):
    """
    Parameters
    ----------
    start: convertable to Datetime
        start range of dates to output
    end: convertable to Datetime
        end range of dates to output
    norm_start : convertable to Datetime or None
        `None` will use in_dataframe.index[0]
    norm_end : convertable to Datetime or None
        if given (not None), output range does not include `norm_end` (it is half-open)
        `None` will use in_dataframe.index[-1]
    freq : {'daily', 'hourly'}
    interp_method : str or None
        `None` will skip resample and interpolation, so
        `in_dataframe` must already be daily or hourly (depending on `freq`)!
    norm_method : {'mean', 'median'}
    """
    if freq == 'hourly':
        hrs = 24
        hrs_freq = '1h'
    elif freq == 'daily':
        hrs = 1
        hrs_freq = '24h'
    else:
        raise ValueError("Invalid `freq` argument value: {}".format(freq))
    if norm_start is None:
        norm_start = in_dataframe.index[0]
    if norm_end is None:
        norm_end = in_dataframe.index[-1]
    else:
        norm_end = pd.to_datetime([norm_end])[0] - pd.Timedelta('1 second')
    print('Computing using range:', norm_start, 'to', norm_end)

    if interp_method is None: # skip resample+interpolation (assumes in_dataframe is daily!)
        t = in_dataframe.loc[norm_start:norm_end]
    else: # resample and interpolate to get hourly
        t = in_dataframe.resample(hrs_freq).interpolate(method=interp_method).loc[norm_start:norm_end]

    if norm_method == 'mean':
        norm = t.groupby([t.index.month, t.index.day, t.index.hour]).mean().sort_index()
    elif norm_method == 'median':
        norm = t.groupby([t.index.month, t.index.day, t.index.hour]).median().sort_index()
    else:
        assert False, "Error: Unknown norm_method '{}'".format(norm_method)

    # now replicate and trim to the desired output range

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # need a non-leapyear and leapyear version
    norm_ly = norm.copy()
    if norm.shape[0] == 366*hrs:
        norm = norm.drop((2,29,))
    else: # norm doesn't include any leapyear data
        assert norm.shape[0] == 365*hrs
        # make Feb 29 the mean of Feb 28 and Mar 1
        foo = (norm.loc[(2,28,)] + norm.loc[(3,1,)]) / 2.0
        foo.index = pd.MultiIndex.from_product( ([2],[29],list(range(hrs))) )
        norm_ly = pd.concat((norm_ly,foo)).sort_index()
        norm_ly.sort_index(inplace=True) # probably not needed

    # build up a 'long normal' (lnorm) dataframe year by year by appending the norm or norm_ly
    lnorm = None
    for yr in np.arange(start.year, end.year+1):
        #print(yr)
        idx = pd.date_range(start='{}-{:02d}-{:02d} {:02d}:00:00'.format(yr,*norm.index[0]),
                            end=  '{}-{:02d}-{:02d} {:02d}:00:00'.format(yr,*norm.index[-1]),
                            freq=hrs_freq)
        if idx.shape[0] == 366*hrs:
            foo = norm_ly.copy()
        else:
            assert norm.shape[0] == 365*hrs
            foo = norm.copy()
        foo.index = idx
        if lnorm is None:
            lnorm = foo
        else:
            lnorm = lnorm.append(foo)

    return lnorm.loc[start:end]
