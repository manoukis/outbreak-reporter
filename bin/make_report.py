#!/usr/bin/env python3
import sys
import os
import time
import argparse
import configparser
from itertools import chain
import logging
import numpy as np
import scipy.interpolate
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re
import glob
from collections import OrderedDict
from distutils.util import strtobool
import subprocess

import temperature_functions

plt.style.use('seaborn-paper')

DEFAULT_LOGGING_LEVEL = logging.INFO
MAX_LOGGING_LEVEL = logging.CRITICAL

def setup_logger(verbose_level):
    fmt=('%(levelname)s %(asctime)s [%(module)s:%(lineno)s %(funcName)s] :: '
            '%(message)s')
    logging.basicConfig(format=fmt, level=max((0, min((MAX_LOGGING_LEVEL,
                        DEFAULT_LOGGING_LEVEL-(verbose_level*10))))))


class CustomArgparseHelpFormatter(argparse.HelpFormatter):
    """Help message formatter for argparse
        combining RawTextHelpFormatter and ArgumentDefaultsHelpFormatter
    """

    def _fill_text(self, text, width, indent):
        return ''.join([indent + line for line in text.splitlines(True)])

    def _split_lines(self, text, width):
        return text.splitlines()

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help


###############################################################


def Main(argv):
    tic_total = time.time()

    # parse cfg_file argument
    conf_parser = argparse.ArgumentParser(description=__doc__,
                                          formatter_class=CustomArgparseHelpFormatter,
                                          add_help=False)  # turn off help so later parse (with all opts) handles it
    conf_parser.add_argument('-c', '--cfg-file', type=argparse.FileType('r'), default='main.cfg',
                             help="Config file specifiying options/parameters.\nAny long option can be set by remove the leading '--' and replace '-' with '_'")
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
    aparser = argparse.ArgumentParser(description=__doc__, parents=[conf_parser],
                                      formatter_class=CustomArgparseHelpFormatter)
    # parse arguments
    aparser.add_argument("--start-year", type=int, help="First year to use historic temperature data for")
    aparser.add_argument("--end-year", type=int,
                         help="Last year to use historic temperature data for.  Defaults to current_year-1")

    aparser.add_argument("--thermal-accumulation-base-temp", type=float,
                         help="Base temperature in 'C for thermal accumulation (degree-day) calculations")
    aparser.add_argument("--DDc_per_generation", type=float,
                         help="Thermal units in DDc for one lifecylce generation")

    aparser.add_argument("--MFP_nR", type=str,
                         help="Number of MEDFOES simulations run for each start date")

    aparser.add_argument('-v', '--verbose', action='count', default=0,
                        help="increase logging verbosity")
    aparser.add_argument('-q', '--quiet', action='count', default=0,
                        help="decrease logging verbosity")
    aparser.set_defaults(**defaults) # add the defaults read from the config file
    args = aparser.parse_args(remaining_argv)

    # check required and optional arguments @TCC TODO more to fill in here
    if not 'basedir' in args or not args.basedir:
        args.basedir = os.getcwd()
    if not 'station_callsign' in args or not args.station_callsign:
        logging.error("station_callsign parameter is required: 4 letter callsign for the weather station to use")
        sys.exit(2)
    if not 'start_year' in args or not args.start_year:
        logging.error("start_year parameter is required")
        sys.exit(2)

    # setup logger
    setup_logger(verbose_level=args.verbose-args.quiet)
    logging.info("Using config file: `{}`".format(cfg_filename))
    logging.info('Using passed arguments: '+str(argv))
    logging.info('args: '+str(args))

    # @TCC TEMP -- print out all the options/arguments
    for k,v in vars(args).items():
        print(k,":",v, file=sys.stderr)

    Fss = [pd.to_datetime(args.SS_start),
           pd.to_datetime(args.SS_F1),
           pd.to_datetime(args.SS_F2),
           pd.to_datetime(args.SS_F3)]

    current_year = Fss[0].year
    START_YEAR = args.start_year
    END_YEAR = current_year-1
    if 'end_year' in args and args.end_year:
        END_YEAR = args.end_year

    ## Load temperature data
    tempdf = temperature_functions.load_temperature_hdf5(
                                        temps_fn="{}_AT_cleaned.h5".format(args.station_callsign),
                                        local_time_offset=args.local_time_offset,
                                        basedir=args.basedir,
                                        start_year=START_YEAR,
                                        truncate_to_full_day=True)


    ###########################################################################

    ## generate DD figures ##
    # group hourly data by day so we can get min and max daily values
    grp = tempdf.groupby(pd.TimeGrouper('D'))
    # actually call the function which computes the degree day generation values
    dd = temperature_functions.compute_BMDD_Fs(grp.min(),
                                               grp.max(),
                                               args.thermal_accumulation_base_temp,
                                               args.DDc_per_generation)

    LOCATION = args.location
    BASEDIR = args.basedir
    datestr = '{}-'+'{:02d}-{:02d}'.format(Fss[0].month, Fss[0].day)
    year_range = np.arange(START_YEAR, END_YEAR+1)
    GEN_DD = args.DDc_per_generation

    ## spagehetti-like plot
    maxdays = np.nanmax([ dd.loc[datestr.format(y)]['F3'] for y in year_range ])
    fig = plt.figure(figsize=(5.5,3.5))
    ax = fig.add_subplot(1,1,1)

    y = current_year
    t = dd.loc[datestr.format(y):pd.to_datetime(datestr.format(y)) + pd.Timedelta(days=maxdays)]
    x = (t.index - t.index[0]).days
    ax.plot(x, t['DD'].cumsum(), '-', c='r', lw=2, label=args.short_name)

    lab = 'previous years'
    for y in year_range:
        t = dd.loc[datestr.format(y):pd.to_datetime(datestr.format(y))+pd.Timedelta(days=maxdays)]
        x = (t.index-t.index[0]).days
        ax.plot(x, t['DD'].cumsum(), '-', c='k', alpha=0.25, label=lab, zorder=1)
        lab = ''

    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    trans2 = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    lab = 'official lifecyle projections'
    for i in range(3):
        y = GEN_DD*(i+1)
        ax.axhline(y=y, c='k', ls=':', alpha=0.5, lw=1)
        ax.text(0, y, ' F{:d}'.format(i+1), transform=trans, ha='left', va='bottom')
        if len(Fss)>i+1:
            x = (Fss[i+1]-Fss[0]).days
            ax.plot([x,x], [0,y], color='c', label=lab)
            lab = ''
            ax.text(x, 0, '{:d}'.format(int(x)), transform=trans2, ha='left', va='bottom')

    ax.set_ylim([0,GEN_DD*3*1.1])
    ax.set_xlim([0, maxdays])

    ax.set_xlabel('days after last fly detection')
    ax.set_ylabel('thermal accumulation [DDc]')
    ax.set_title(LOCATION+' $\degree D$ gen from '+datestr.format(current_year))

    # ax.legend(loc='lower right')
    # fig.tight_layout()
    leg = ax.legend(ncol=3, bbox_to_anchor=(-.10,-1.18, 1.1, 1), loc='upper center')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(os.path.join(BASEDIR, 'thermal_accumulation.pdf'), bbox_inches='tight', pad_inches=0)

    ## historic DD lifecylce histograms
    fig = plt.figure(figsize=(5.5,3.5))
    ax0 = None
    for Fnum in range(1,4):
        fall = np.array([ dd.loc[datestr.format(y)]['F'+str(Fnum)] for y in year_range ])
        if ax0 is None:
            ax = fig.add_subplot(3,1,Fnum)
            ax0 = ax
        else:
            ax = fig.add_subplot(3,1,Fnum, sharex=ax0)

        ax.hist(fall[~np.isnan(fall)], label='previous years', normed=True)

        if len(Fss) > Fnum:
            ax.axvline(x=(Fss[Fnum]-Fss[0]).days, color='c', label='official lifecyle projections')
        ax.set_xlabel('F{} [days]'.format(Fnum))
        if Fnum<3:
            plt.setp(ax.get_xticklabels(), visible=False)

    ax0.set_title(LOCATION+' $\degree D$ gen from '+datestr.format(current_year))
    h, l = ax0.get_legend_handles_labels()
    fig.legend(h,l, bbox_to_anchor=(0.5, .035), loc='center', ncol=3, borderaxespad=0)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(os.path.join(BASEDIR, 'DD_previous_years_lifecycles_histograms.pdf'), bbox_inches='tight', pad_inches=0)


    ### MEDFOES PE Quantiles from results ###
    # used for summary table and plot
    MEDFOES_DIR = os.path.join(args.basedir, args.medfoes_dir)
    QUANTILES_TO_CALC = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
    datestr = '{}-' + '{:02d}-{:02d}'.format(Fss[0].month, Fss[0].day)
    year_range = np.arange(START_YEAR, END_YEAR + 1)
    if not current_year in year_range:
        year_range = np.append(year_range, current_year)

    pe = pd.DataFrame(columns=QUANTILES_TO_CALC)
    for i, y in enumerate(year_range):
        mfrunset = datestr.format(y)
        date = pd.to_datetime(mfrunset)
        detail_fn = glob.glob(os.path.join(MEDFOES_DIR, 'runs', mfrunset, 'MED-FOESp_*_detail.txt'))
        if len(detail_fn) == 0:
            logging.warn("No MEDFOES runs found for '{}'".format(mfrunset))
            continue
        assert len(detail_fn) == 1
        detail_fn = detail_fn[0]
        assert detail_fn
        d = pd.read_csv(detail_fn, sep='\t')
        # set any runs which didn't fully complete to inf
        d.loc[(d['end_condition']!=0) | (d['end_flies']!=0), 'run_time'] = np.inf
        pe.loc[date] = d['run_time'].quantile(QUANTILES_TO_CALC, interpolation='linear')


    ## Tablular summary results ##
    datestr = '{}-'+'{:02d}-{:02d}'.format(Fss[0].month, Fss[0].day)
    year_range = np.arange(START_YEAR, END_YEAR+1)
    f = []
    fcur = []
    for Fnum in range(1,4):
        x = pd.Series([ dd.loc[datestr.format(y)]['F{:d}'.format(Fnum)] for y in year_range ],
                     index=pd.to_datetime([datestr.format(y) for y in year_range]))
        x.dropna(inplace=True)
        f.append(x)
        fcur.append(dd.loc[datestr.format(current_year)]['F{:d}'.format(Fnum)])
    # Add medfoes 95% exradication result too
    f.append(pe.loc[datestr.format(START_YEAR):datestr.format(END_YEAR)][0.95]/24.0)
    try:
        current_pe95 = pe.loc[datestr.format(current_year)][0.95] / 24.0
    except KeyError:
        current_pe95 = np.nan

    tab = pd.DataFrame(OrderedDict([
            [('','official projections'), ["{:.0f}".format((Fss[x]-Fss[0]).days) for x in range(1,4)]+["-"]],
            [('',args.short_name), ["{:.0f}".format(x) if np.isfinite(x) else "-" for x in fcur]
                                  +["{:.1f}".format(current_pe95) if np.isfinite(current_pe95) else "-"]],
            [('historic','25\%'), ["{:.1f}".format(x.quantile(0.25)) for x in f]],
            [('historic','50\%'), ["{:.1f}".format(x.quantile(0.5)) for x in f]],
            [('historic','75\%'), ["{:.1f}".format(x.quantile(0.75)) for x in f]],
            [('historic','mean'), ["{:.1f}".format(x.mean()) for x in f]],
            [('historic','std'), ["{:.1f}".format(x.std()) for x in f]],
            [('historic','years (N)'), ["{:d}".format(int(x.count())) for x in f]],
            ]), index=[r'\makecell{DD F1 \\ days}',
                       r'\makecell{DD F2 \\ days}',
                       r'\makecell{DD F3 \\ days}',
                       r'\makecell{ABS \\ 95\% erad}'] ).T
    tab.index = pd.MultiIndex.from_tuples(tab.index)
    #pd.set_option('precision', 1)
    tab.to_latex(os.path.join(BASEDIR, "summary_table.texi"), na_rep='-',
                 bold_rows=False, multirow=True, column_format='|lr|rrrr|', escape=False)


    ## Results latex include file
    with open(os.path.join(BASEDIR,"result_variables.texi"), 'w') as fh:
        for name, val in (
                ('ShortName', args.short_name),
                ('OutbreakLocation', args.location),
                ('StationDescription', args.station_description),
                ('Station', args.station_callsign),
                ('TempStartYear', "{:04d}".format(START_YEAR)),
                ('TempEndYear', "{:04d}".format(END_YEAR)),
                ('StartDate', Fss[0].strftime("%Y-%m-%d")),
                ('StartDOY', Fss[0].strftime("%m-%d")),
                ('SSFA', Fss[1].strftime("%Y-%m-%d")),
                ('SSFB', Fss[2].strftime("%Y-%m-%d")),
                ('SSFC', Fss[3].strftime("%Y-%m-%d")),
                # ('SSFADays', str((Fss[1]-Fss[0]).days)),
                # ('SSFBDays', str((Fss[2]-Fss[0]).days)),
                # ('SSFCDays', str((Fss[3]-Fss[0]).days)),
                ('EndDateOfTempData', tempdf.index[-1].strftime("%Y-%m-%d")),
                ('DDBaseTempC', "{:.4f}".format(args.thermal_accumulation_base_temp)),
                ('DDBaseTempF', "{:.4f}".format((args.thermal_accumulation_base_temp*9/5)+32)),
                ('GenDDc', "{:.4f}".format(args.DDc_per_generation)),
                ('GenDDf', "{:.4f}".format(args.DDc_per_generation*9/5)),
                ('LargeGapSize', args.potentially_problematic_gap_size),
                ('MEDFOESVersion', re.search(r'MedFoesP-(.*).jar',
                                             glob.glob(os.path.join(MEDFOES_DIR,'MedFoesP*.jar'))[0]).group(1)),
                ('MFPnR', args.MFP_nR),
        ):

            print(r"\newcommand{\Var"+name+r"}{"+val+r"\xspace}", file=fh)


    ## Summary plot ##
    fig = plt.figure(figsize=(5.5,3.5))
    ax = fig.add_subplot(1,1,1)

    datestr = '{}-'+'{:02d}-{:02d}'.format(Fss[0].month, Fss[0].day)
    year_range = np.arange(START_YEAR, END_YEAR+1)

    fall = pd.Series([ dd.loc[datestr.format(y)]['F3'] for y in year_range ],
                     index=pd.to_datetime([datestr.format(y) for y in year_range]))
    fall.dropna(inplace=True)

    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    ax.plot(fall.index, fall, ls='none', marker='o', mfc='none', mec='C0', mew=1, label='degree-day F3')
    y = np.median(fall)
    ax.axhline(y=y, ls='--', color='C0', label='median degree-day F3')
    ax.text(0, y, '{:d}'.format(int(np.round(y))), transform=trans, ha='left', va='bottom')
    cdate = datestr.format(current_year)
    ax.plot(dd.loc[cdate].name, dd.loc[cdate]['F3'], ls='none', marker='o', mfc='C0', mec='C0', mew=1)

    tmp = sorted([datestr.format(x) for x in year_range])
    hist_pe = pe[tmp[0]:tmp[-1]]
    ax.plot(hist_pe[0.95]/24, ls='none', marker='d', mfc='none', mec='C1', mew=1, label='MED-FOES 95% erad.')
    y = np.median(hist_pe[0.95]/24)
    ax.axhline(y=y, ls=':', color='C1', label='median MED-FOES 95% erad.')
    if y > 0:
        ax.text(0, y, '{:d}'.format(int(np.round(y))), transform=trans, ha='left', va='bottom')
    try:
        current_pe = pe.loc[datestr.format(current_year)]
        ax.plot(current_pe.name, current_pe[0.95]/24.0, ls='none', marker='d', mfc='C1', mec='C1', mew=1)
    except KeyError:
        pass

    y = (Fss[3]-Fss[0]).days
    ax.axhline(y=y, color='r', label='official F3 value')
    ax.text(1, y, '{:d}'.format(int(np.round(y))), transform=trans, ha='left', va='center')

    ax.set_ylabel('days after start date')
    ax.set_xlabel('start date')
    ax.set_title(LOCATION+" PQL based on "+datestr.format(current_year))
    leg = ax.legend(ncol=3, bbox_to_anchor=(-.10,-.4, 1.1, 1), loc='lower center')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(os.path.join(BASEDIR, 'summary_plot.pdf'), bbox_inches='tight', pad_inches=0)


    ## MEDFOES spaghetti plot
    MEDFOES_DIR = os.path.join(args.basedir, args.medfoes_dir)
    medfoes_runs_per_date = int(args.MFP_nR)
    datestr = '{}-' + '{:02d}-{:02d}'.format(Fss[0].month, Fss[0].day)
    year_range = np.append(np.arange(START_YEAR, END_YEAR + 1), current_year)
    if not current_year in year_range:
        year_range = np.append(year_range, current_year)
    fig = plt.figure(figsize=(5.5,3.5))
    ax = fig.add_subplot(1, 1, 1)

    lab = 'previous years'
    maxdays = 0
    for i, year in enumerate(year_range):
        mfrunset = datestr.format(year)
        date = pd.to_datetime(mfrunset)
        detail_fn = glob.glob(os.path.join(MEDFOES_DIR, 'runs', mfrunset, 'MED-FOESp_*_detail.txt'))
        if len(detail_fn) == 0:
            logging.warn("No MEDFOES runs found for '{}'".format(mfrunset))
            continue
        assert len(detail_fn) == 1
        detail_fn = detail_fn[0]
        assert detail_fn
        # print(detail_fn)
        d = pd.read_csv(detail_fn, sep='\t')

        y = d[d['end_condition'] == 0]['run_time'].sort_values()
        y.index = np.arange(1, len(y) + 1) / medfoes_runs_per_date
        tmp = y.max() / 24.0
        if maxdays < tmp:
            maxdays = tmp

        if year == current_year:
            ax.plot(y / 24.0, y.index, '-', c='r', lw=1.5, label=args.short_name)
        else:
            ax.plot(y / 24.0, y.index, '-', c='k', alpha=0.25, label=lab)
            lab = ''

    trans = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    lab = 'official lifecyle projections'
    for i in range(3):
        y = GEN_DD * (i + 1)
        ax.axhline(y=y, c='k', alpha=0.5, lw=.5)
        if len(Fss) > i + 1:
            x = (Fss[i + 1] - Fss[0]).days
            ax.plot([x, x], [0, y], color='c', label=lab)
            lab = ''
            ax.text(x, 0, '{:d}'.format(int(x)), transform=trans, ha='left', va='bottom')
            ax.text(x, .99, 'F{:d}'.format(i + 1), transform=trans, ha='right', va='top')

    ax.axhline(y=0.95, ls=':', lw=1, c='k', alpha=0.5, label='95% eradication threshold')
    # trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    # ax.text(0, 0.95, '95\% erad', transform=trans, ha='right', va='top')

    ax.set_ylim([0, 1])
    ax.set_xlim([0, maxdays])

    ax.set_xlabel('days after last fly detection')
    ax.set_ylabel('MEDFOES prop. runs eradicated')
    ax.set_title(LOCATION + ' MEDFOES from ' + datestr.format(current_year))

    # ax.legend(loc='lower right')
    # fig.tight_layout()
    leg = ax.legend(ncol=2, bbox_to_anchor=(-.10,-1.18, 1.1, 1), loc='upper center')
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(os.path.join(BASEDIR, 'ABS_previous_years_PQL.pdf'), bbox_inches='tight', pad_inches=0)


    ## Run pdflatex
    logging.info("Compiling latex report")
    p = subprocess.Popen([os.path.join(args.basedir,'latex_report.sh')], cwd=args.basedir)
    p.wait()

    ## end
    logging.info("Done: {:.2f} sec elapsed".format(time.time()-tic_total))
    return 0




#########################################################################
# Main loop hook... if run as script run main, else this is just a module
if __name__ == "__main__":
    sys.exit(Main(argv=None))
