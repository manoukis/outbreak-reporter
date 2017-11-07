#!/usr/bin/env python3
import os
import sys
import logging
import time
import argparse
import configparser
from itertools import chain
from distutils.util import strtobool
import numpy as np
import scipy.interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import ftplib
import io
import datetime
import dateutil
from collections import OrderedDict
import temperature_functions

plt.style.use('seaborn-paper')

DEFAULT_LOGGING_LEVEL = logging.INFO
MAX_LOGGING_LEVEL = logging.CRITICAL

ISD_LITE_FTP_BASE_PATH = "/pub/data/noaa/isd-lite"
ISD_LITE_SPEC = [
    [( 0, 4), 'year', int],
    [( 5, 7), 'month', int],
    [( 8,10), 'day', int],
    [(11,13), 'hour', int],
    [(13,19), 'air temp', lambda x: np.nan if x.strip()=='-9999' else int(x)/10],
    [(19,25), 'dewpoint', lambda x: np.nan if x.strip()=='-9999' else int(x)/10],
    [(25,31), 'sea level pressure', lambda x: np.nan if x=='-9999' else int(x)/10],
    [(31,37), 'wind direction', lambda x: np.nan if x=='-9999' else int(x)],
    [(37,43), 'wind speed', lambda x: np.nan if x=='-9999' else int(x)/10],
    [(43,49), 'sky condition total coverage code', lambda x: np.nan if x=='-9999' else int(x)],
    [(49,55), 'liquid percip 1hr', lambda x: np.nan if x=='-9999' else int(x)/10],
    [(55,61), 'liquid percip 6hr', lambda x: np.nan if x=='-9999' else int(x)/10],
    ]


def Main(argv):
    tic_total = time.time()

    # parse cfg_file argument
    conf_parser = argparse.ArgumentParser(description=__doc__,
                                          add_help=False)  # turn off help so later parse (with all opts) handles it
    conf_parser.add_argument('-c', '--cfg-file', type=argparse.FileType('r'), default='main.cfg',
                             help="Config file specifiying options/parameters.\nAny long option can be set by removing the leading '--' and replacing '-' with '_'")
    args, remaining_argv = conf_parser.parse_known_args(argv)
    # build the config (read config files)
    cfg_filename = None
    if args.cfg_file:
        cfg_filename = args.cfg_file.name
        cfg = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
        cfg.optionxform = str # make configparser case-sensitive
        cfg.read_file(chain(("[DEFAULTS]",), args.cfg_file))
        defaults = dict(cfg.items("DEFAULTS"))
        # special handling of paratmeters that need it like lists
        # defaults['make_temperature_plots'] = strtobool(defaults['make_temperature_plots'])
        #        if( 'bam_files' in defaults ): # bam_files needs to be a list
        #            defaults['bam_files'] = [ x for x in defaults['bam_files'].split('\n') if x and x.strip() and not x.strip()[0] in ['#',';'] ]
    else:
        defaults = {}

    # Parse rest of arguments with a new ArgumentParser
    aparser = argparse.ArgumentParser(description=__doc__, parents=[conf_parser])
    # parse arguments
    aparser.add_argument('-f', '--force', action='store_true', default=False,
                         help="Download/fetch temperature data instead of just updating")
    aparser.add_argument('-P', '--make-temperature-plots', type=strtobool, default=False,
                         help="Generate plots when fetching and cleaning temperatures")
    aparser.add_argument("--start-year", type=int, help="First year to use historic temperature data for")
    aparser.add_argument("--potentially-problematic-gap-size", type=pd.Timedelta,
                         help="Time gaps in temperature data longer than this are noted and filled via day-over-day interpolation")
    aparser.add_argument("--outlier_rolling_sigma_window", type=int, help="0 to just use median instead of median/sigma")
    aparser.add_argument("--outlier_rolling_median_window", type=int, help="Size of window for diff from median (or median/sigma) outlier detection")
    aparser.add_argument("--outlier_thresh", type=float, help="Threshold for outlier detection")
    aparser.add_argument("--outlier_multipass", type=strtobool, help="Do multiple outlier removal passes; use True")

    aparser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase logging verbosity")
    aparser.add_argument('-q', '--quiet', action='count', default=0,
                        help="decrease logging verbosity")
    aparser.set_defaults(**defaults) # add the defaults read from the config file
    args = aparser.parse_args(remaining_argv)

    # check required arguments @TCC TODO more to fill in here
    if not 'basedir' in args or not args.basedir:
        args.basedir = os.getcwd()
    if not 'station_callsign' in args or not args.station_callsign:
        logging.error("station_callsign parameter is required: 4 letter callsign for the weather station to use")
        sys.exit(2)
    if not 'start_year' in args or not args.start_year:
        args.start_year = None

    # setup logger
    setup_logger(verbose_level=args.verbose-args.quiet)
    logging.info("Using config file: `{}`".format(cfg_filename))
    logging.info('Using passed arguments: '+str(argv))
    logging.info('args: '+str(args))

    # # @TCC TEMP -- print out all the options/arguments
    # for k,v in vars(args).items():
    #     print(k,":",v, file=sys.stderr)

    datadir = os.path.join(args.basedir, args.temperature_data_dir)

    ####
    tempdf = fetch_and_process_temperatures(
                                    station_callsign=args.station_callsign,
                                    start_year=args.start_year,
                                    basedir=args.basedir,
                                    datadir=os.path.join(args.basedir, args.temperature_data_dir),
                                    make_temperature_plots=args.make_temperature_plots,
                                    NOAA_ISD_ftp_host=args.NOAA_ISD_ftp_host,
                                    potentially_problematic_gap_size=args.potentially_problematic_gap_size,
                                    force_download_temps=args.force,
                                    outlier_rolling_sigma_window=args.outlier_rolling_sigma_window,
                                    outlier_rolling_median_window=args.outlier_rolling_median_window,
                                    outlier_thresh=args.outlier_thresh,
                                    outlier_multipass=args.outlier_multipass,
                                )

    # save as a csv with proper timezone
    outfn = "{}_AT_cleaned".format(args.station_callsign)
    logging.info("Saving csv format to '{}'".format(outfn))
    tempdf.index = tempdf.index.tz_convert(
                    temperature_functions.offset_str_to_tz(args.local_time_offset))
    tempdf.to_csv(os.path.join(args.basedir, outfn+'.csv'), columns=['AT'], index_label='datetime')

    ## end
    logging.info("Done: {:.2f} sec elapsed".format(time.time()-tic_total))
    return 0


def fetch_and_process_temperatures(station_callsign,
                                   start_year,
                                   basedir,
                                   datadir,
                                   make_temperature_plots,
                                   NOAA_ISD_ftp_host,
                                   potentially_problematic_gap_size,
                                   force_download_temps,
                                   outlier_rolling_sigma_window=24*5,  # None or 0 to just use median instead of median/sigma
                                   outlier_rolling_median_window=5,
                                   outlier_thresh=1.5,  # deviation from media/sigma to trigger removal
                                   outlier_multipass=True,  # cycle until no points removed, or False for not
                                   ):
    os.makedirs(datadir, exist_ok=True)
    now_year = datetime.datetime.now().year

    # get the isd_history (aka station_info)
    station_info = get_ISD_history_file(datadir, NOAA_ISD_ftp_host, save_file=True,
                                     force_download=force_download_temps)
    # pick just the info associated with the station we want
    station_info = station_info[station_info['CALL'] == station_callsign]
    logging.info("Station info for "+station_callsign+"\n"+str(station_info))

    # load the last fetch timestamp
    last_fetch_time = None
    if not force_download_temps:
        try:
            with open(os.path.join(datadir, "last_fetch_time.txt"), 'r') as fh:
                last_fetch_time = fh.readline().strip()
                last_fetch_time = time.strptime(last_fetch_time, "%Y-%m-%dT%H:%M:%S%z")
        except FileNotFoundError:
            logging.info("No last_fetch_time.txt file found")
            last_fetch_time = None
            pass
        except ValueError as err:
            logging.error("Did not understand last_fetch_time.txt: "+str(err))
            last_fetch_time = None

    isd_files = []
    to_get = []
    for k, row in station_info.iterrows():
        for year in range(int(row['BEGIN'][0:4]), int(row['END'][0:4])+2):  # +2 because isd-history may be out of date
            if year <= now_year:
                isd_files.append([year, "{}-{}-{:04d}.gz".format(row['USAF'], row['WBAN'], year)])
                if last_fetch_time is None or year >= last_fetch_time.tm_year:
                    to_get.append(isd_files[-1])

    logging.info("Number of possible ISD files = {:d}".format(len(isd_files)))
    logging.info("Number of files try and get/update = {:d}".format(len(to_get)))

    ## Download isd-lite files
    # print(*to_get, sep="\n")
    if len(to_get) > 0:
        with ftplib.FTP(host=NOAA_ISD_ftp_host) as ftpconn:
            ftpconn.login()
            for year, fn in to_get:
                url = "{}/{:04d}/{}".format(ISD_LITE_FTP_BASE_PATH,year,fn)
                logging.info("Fetching: '{}'".format(url))
                has_err = False
                with open(os.path.join(datadir, fn), 'wb') as fh:
                    try:
                        ftpconn.retrbinary('RETR ' + url, fh.write)
                    except ftplib.error_perm as err:
                        has_err = True
                        if str(err).startswith('550 '):
                            logging.warning(err)
                        else:
                            raise
                if has_err:
                    os.remove(os.path.join(datadir, fn))

    # save the timestamp to a file
    with open(os.path.join(datadir, "last_fetch_time.txt"), 'w') as fh:
        print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), file=fh)

    ## Load the isd-lite data from files
    tempdf = None
    for year, fn in isd_files:
        print(fn)
        try:
            d = pd.read_fwf(os.path.join(datadir, fn),
                            colspecs=[x[0] for x in ISD_LITE_SPEC],
                            names=[x[1] for x in ISD_LITE_SPEC],
                            converters=OrderedDict([[x[1], x[2]] for x in ISD_LITE_SPEC]),
                            compression='infer')
            if tempdf is None:
                tempdf = d
            else:
                tempdf = pd.concat([tempdf, d])
        except FileNotFoundError:
            # logging.warning("File not found: '{}'".format(fn))
            pass
    # make a datetime index
    tempdf.index = pd.to_datetime(tempdf[['year', 'month', 'day', 'hour']])
    tempdf.index = tempdf.index.tz_localize('UTC')
    # sort
    tempdf.sort_index(inplace=True)

    # extract just air temp and rename columns
    t = pd.DataFrame(tempdf['air temp'])
    t.columns = ['AT']
    t.index.rename('datetime', inplace=True)

    if make_temperature_plots:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(t.index, t['AT'], '.', label='raw temperatures')
        ax.set_title(station_callsign)
        ax.set_xlabel("datetime")
        ax.set_ylabel("air temp ['C]")
        plt.savefig(os.path.join(basedir, '{}_AT_orig.png'.format(station_callsign)))

    ### Cleaning temperatures ###

    ## deduplication (should not be needed with ISD-lite data)
    tmp = t[t.index.duplicated(keep=False)].sort_index()
    if len(tmp) > 0:
        logging.warning("{} duplicate temperature entries found".format(len(tmp)))
        t = t[~t.index.duplicated(keep='first')].sort_index()

    ## drop nan values
    tmp = t.shape[0]
    t.dropna(inplace=True)
    logging.info("Dropping {} NaN values".format(tmp-t.shape[0]))
    # keep a copy of the original temperatures for final plot if needed
    if make_temperature_plots:
        ot = t.copy(deep=True)

    ## Outlier removal
    cum_num = 0
    while True:
        if outlier_rolling_sigma_window > 0:
            sigma = t['AT'].rolling(window=outlier_rolling_sigma_window, center=True).std()
        else:
            sigma = 1
        diff = (t['AT']-t['AT'].rolling(window=outlier_rolling_median_window, center=True).median())/sigma
        outlier_mask = diff.abs() > outlier_thresh
        num = np.count_nonzero(outlier_mask)
        cum_num += num
        if num == 0:
            break
        else:
            logging.info("Outlier removal of {} points".format(num))
        t = t[~outlier_mask]
        if not outlier_multipass:
            break

    # plot showing what is being removed
    if make_temperature_plots:
        if cum_num > 0:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax = ot[~ot.index.isin(t.index)].plot(ax=ax, linestyle='none', marker='o', color='r', zorder=8)
            # ax = ot.plot(ax=ax, linestyle='-', linewidth=1, marker=None, color='red')
            ax = t.plot(ax=ax, linestyle='none', marker='.', color='blue')
            ax.legend(['outlier', 'original', 'cleaned'])
            ax.set_title(station_callsign)
            ax.set_ylabel("air temp ['C]")
            fn = '{}_outlier.png'.format(station_callsign)
            fig.savefig(os.path.join(basedir, fn))

    ## "by-hand" fixes for individual datasets
    # @TCC TODO MAKE THIS PART OF CONFIG
    if station_callsign == 'KSNA': # KSNA (Orange County)
        # 2016-08-14 to 2016-08-15 overnight has some >0 values when they should be more like 19-20
        remove_spurious_temps(t, '< 0', '2016-08-14', '2016-08-15', inplace=True)
    if station_callsign == 'KSFO':
        remove_spurious_temps(t, '< 0', '1976-07-16', '1976-07-17', inplace=True)
    if station_callsign == 'KRIV':
        remove_spurious_temps(t, '< 0', '1995-11-15', '1995-11-15', inplace=True)

    ## Larger gap identification
    gaps_filename = os.path.join(basedir, "{}_AT_gaps.tsv".format(station_callsign))
    gaps = t.index.to_series().diff()[1:]  # @TCC had ot instead of t for some reason
    idx = np.flatnonzero(gaps > potentially_problematic_gap_size)
    prob_gaps = gaps[idx]
    # save to file for future reference
    with open(gaps_filename,'w') as fh:
        # output the gaps, biggest to smallest, to review
        print('#', station_callsign, t.index[0].isoformat(), t.index[-1].isoformat(), sep='\t', file=fh)
        print('# Potentially problematic gaps:', len(prob_gaps), file=fh)
        logging.info('Potentially problematic gaps: {}'.format(len(prob_gaps)))
        tmp = prob_gaps.sort_values(ascending=False)
        for i in range(len(tmp)):
            rng = [tmp.index[i]-tmp.iloc[i], tmp.index[i]]
            print(rng[0], rng[1], rng[1]-rng[0], sep='\t', file=fh)
    newidx = pd.date_range(start=t.index[0].round('d')+pd.Timedelta('0h'),
                           end=t.index[-1].round('d')-pd.Timedelta('1s'),
                           freq='1h', tz='UTC')
    # Simple linear interpolation
    at_interp_func = scipy.interpolate.interp1d(t.index.astype('int64').values,
                                           t['AT'].values,
                                           kind='linear',
                                           fill_value=np.nan, #(0,1)
                                           bounds_error=False)
    nt = pd.DataFrame({'AT':at_interp_func(newidx.astype('int64').values)},
                       index=newidx)
    # Fill those gaps using day-to-day (at same hour) interpolation
    gap_pad = pd.Timedelta('-10m') # contract the gaps a bit so we don't remove good/decent edge values
    t = nt.copy(deep=True) # operate on a copy so we can compare with nt
    # fill the gap ranges with nan (replacing the default interpolation)
    for i in range(len(prob_gaps)):
        rng = [prob_gaps.index[i]-prob_gaps.iloc[i], prob_gaps.index[i]]
        t[rng[0]-gap_pad:rng[1]+gap_pad] = np.nan
    # reshape so each row is a whole day's (24) data points
    rows = int(t.shape[0]/24)
    foo = pd.DataFrame(t.iloc[:rows*24].values.reshape((rows,24)))
    # simple linear interpolation
    foo.interpolate(metnod='time', limit=24*60, limit_direction='both', inplace=True)
    # reshape back
    t = pd.DataFrame({'AT':foo.stack(dropna=False).values}, index=t.index[:rows*24])

    # save final cleaned temperatures
    outfn = "{}_AT_cleaned".format(station_callsign)
    print("Saving cleaned temp data to:",outfn)
    t.to_hdf(os.path.join(basedir, outfn+'.h5'), 'table', mode='w',
              data_colums=True, complevel=5, complib='bzip2',
              dropna=False)

    # plot temperatures
    if make_temperature_plots:
        r1 = t.index[0]
        r2 = t.index[-1]
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(ot.loc[r1:r2].index, ot.loc[r1:r2]['AT'], linestyle='none', marker='.', label='raw')
        ax.plot(nt.loc[r1:r2].index, nt.loc[r1:r2]['AT'], linestyle='-', marker=None, lw=1, label='interpolated')
        for i in range(len(prob_gaps)):
            if i == 0: # only label first segment
                label = 'filled'
            else:
                label = ''
            rng = [tmp.index[i]-tmp.iloc[i], tmp.index[i]]
            ax.plot(t.loc[rng[0]:rng[1]].index, t.loc[rng[0]:rng[1]]['AT'], '.-', lw=1, color='r', label=label)
        ax.set_xlim((r1,r2))
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Temperature [$\degree$C]')
        ax.set_title(station_callsign)
        ax.legend()
        fig.savefig(os.path.join(basedir, '{}_cleaning.png'.format(station_callsign)))

    return t



############################################################3

def setup_logger(verbose_level):
    fmt=('%(levelname)s %(asctime)s [%(module)s:%(lineno)s %(funcName)s] :: '
            '%(message)s')
    logging.basicConfig(format=fmt, level=max((0, min((MAX_LOGGING_LEVEL,
                        DEFAULT_LOGGING_LEVEL-(verbose_level*10))))))


def get_ISD_history_file(datadir, NOAA_ISD_ftp_host, save_file=True, force_download=False):
    ## get (fetch or load cached) isd-history.txt file
    tmp = None
    if not force_download: # try to load cached isd-history file
        try:
            tmp = read_isd_history_stations_list(os.path.join(datadir, 'isd-history.txt'))
        except FileNotFoundError:
            pass
    if tmp is None: # fetch
        logging.info("Fetching isd-history.txt")
        with ftplib.FTP(host=NOAA_ISD_ftp_host) as ftpconn:
            ftpconn.login()
            ftp_file = "/pub/data/noaa/isd-history.txt"
            # read the whole file and save it to a BytesIO (stream)
            response = io.BytesIO()
            try:
                ftpconn.retrbinary('RETR ' + ftp_file, response.write)
            except ftplib.error_perm as err:
                if str(err).startswith('550 '):
                    logging.error(err)
                else:
                    raise
        if save_file:
            response.seek(0)  # jump back to the beginning of the stream
            with open(os.path.join(datadir, 'isd-history.txt'),'wb') as fh:
                fh.write(response.getvalue())
        response.seek(0)  # jump back to the beginning of the stream
        tmp = read_isd_history_stations_list(response)
    return tmp


def read_isd_history_stations_list(filename, skiprows=22):
    """Read and parse stations information from isd_history.txt file"""
    fwfdef = (( ('USAF', (6, str)),
                ('WBAN', (5, str)),
                ('STATION NAME', (28, str)),
                ('CTRY', (4, str)),
                ('ST', (2, str)),
                ('CALL', (5, str)),
                ('LAT', (7, str)),
                ('LON', (8, str)),
                ('EVEV', (7, str)),
                ('BEGIN', (8, str)),
                ('END', (8, str)),
                ))
    names = []
    colspecs = []
    converters = {}
    i = 0
    for k,v in fwfdef:
        names.append(k)
        colspecs.append((i, i+v[0]+1))
        i += v[0]+1
        converters[k] = v[1]
    stdf = pd.read_fwf(filename, skiprows=skiprows,
                       names=names,
                       colspecs=colspecs,
                       converters=converters)
    return stdf


def remove_spurious_temps(ot, query_op, date1, date2=None, plot=True, inplace=False):
    if date2 is None:
        date2 = date1
    #ax = ot.loc[date1:date2].plot(ax=None, linestyle='-', marker='o') # plot
    out_t = ot.drop(ot.loc[date1:date2].query('AT {}'.format(query_op)).index, inplace=inplace)
    if inplace:
        out_t = ot
    #out_t.loc[date1:date2].plot(ax=ax, linestyle='-', marker='*') # plot'
    #ax.set_title("Remove AT {}, range=[{}:{}]".format(query_op, date1, date2))
    return out_t


#########################################################################
# Main loop hook... if run as script run main, else this is just a module
if __name__ == "__main__":
    sys.exit(Main(argv=None))
