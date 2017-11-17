#!/usr/bin/env python3
import sys
import os
import time
import argparse
import configparser
from itertools import chain
import logging
import numpy as np
import pandas as pd
from distutils.util import strtobool

import temperature_functions


MFP_CFG_PAT = """
Ni      33,100
R       2
S       0.05,0.15
rred    .5,1
Sai     true
TEL     9.6,12.5,27.27,33.80
TLP     5.0,10.8,94.50,186.78
TPA     9.1,13.8,123.96,169.49
TIM     7.9,9.9,58.20,105.71
Me      0.0198,0.1211
Ml      0.0068,0.0946
Mp      0.0016,0.0465
Ma      0.0245,0.1340
tdm     true
r       5,35
rvar    3.57
Dm      1
TuSD    0.05
Tmax    35
nR      {NUM_RUNS_PER_SET}
nT      20
Mx      500000
seed    1
T       dummy_value_overridden_by_command_line_argument
o       dummy_value_overridden_by_command_line_argument
q       true
pr      false
plot    false
"""


DEFAULT_LOGGING_LEVEL = logging.INFO
MAX_LOGGING_LEVEL = logging.CRITICAL

def setup_logger(verbose_level):
    fmt=('%(levelname)s %(asctime)s [%(module)s:%(lineno)s %(funcName)s] :: '
            '%(message)s')
    logging.basicConfig(format=fmt, level=max((0, min((MAX_LOGGING_LEVEL,
                        DEFAULT_LOGGING_LEVEL-(verbose_level*10))))))

###############################################################


def Main(argv):
    tic_total = time.time()

    # parse cfg_file argument
    conf_parser = argparse.ArgumentParser(description=__doc__,
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
    aparser = argparse.ArgumentParser(description=__doc__, parents=[conf_parser])

    # parse arguments
    aparser.add_argument("--start-year", type=int, help="First year to use historic temperature data for")
    aparser.add_argument("--end-year", type=int,
                         help="Last year to use historic temperature data for.  Defaults to current_year-1")

    aparser.add_argument('-f', "--force", action='store_true', default=False,
                         help="Update all input temperature files and rerun MEDFOES for all years")
    aparser.add_argument("--medfoes-num-years-of-temp-data-per-set", type=int,
                         help="How many years (int) worth of temperature data for each MEDFOES simulations run set.  Increase if runs not going to extirpation")

    # MEDFOES config
    aparser.add_argument("--MFP-Ni", default='33,100',
                         help="Estimated number of adult females in the initial population (LHS range)")
    aparser.add_argument("--MFP-Ad", default='29.8,49.7,15.6,1.8228,3.0772',
                         help="Initial age distribution (egg to mature adult) default from Carey 1982")
    aparser.add_argument("--MFP-R", default='2',
                         help="Number of days between simulation start and beginning of interventions")
    aparser.add_argument("--MFP-S", default='0.05,0.15',
                         help="Daily human-induced adult mortality from interventions (LHS range)")
    aparser.add_argument("--MFP-rred", default='0.5,1',
                         help="Loss of reproductive flies due to intervention (SIT) (LHS range)")
    aparser.add_argument("--MFP-Sai", default='true',
                         help="New flies are not reproductive after intervention start (innundative SIT)")
    aparser.add_argument("--MFP-TEL", default='9.6,12.5,27.27,33.80',
                         help="Stage transition parameters, egg to larvae (LHS ranges)")
    aparser.add_argument("--MFP-TLP", default='5.0,10.8,94.50,186.78',
                         help="Stage transition parameters, larvae to pupae (LHS ranges)")
    aparser.add_argument("--MFP-TPA", default='9.1,13.8,123.96,169.49',
                         help="Stage transition parameters, pupae to immature adult (LHS ranges)")
    aparser.add_argument("--MFP-TIM", default='7.9,9.9,58.20,105.71',
                         help="Stage transition parameters, immature to mature adult (LHS ranges)")
    aparser.add_argument("--MFP-Me", default='0.0198,0.1211',
                         help="Daily natural mortality of eggs (LHS range)")
    aparser.add_argument("--MFP-Ml", default='0.0068,0.0946',
                         help="Daily natural mortality of larvae (LHS range)")
    aparser.add_argument("--MFP-Mp", default='0.0016,0.0465',
                         help="Daily natural mortality of pupae (LHS range)")
    aparser.add_argument("--MFP-Ma", default='0.0245,0.1340',
                         help="Daily natural mortality of adults (LHS range)")
    aparser.add_argument("--MFP-tdm", default='true',
                         help="Temperature affects mortality")
    aparser.add_argument("--MFP-r", default='5,35',
                         help="Eggs produced per reproduction event (LHS range)")
    aparser.add_argument("--MFP-rvar", default='3.57',
                         help="Variance in eggs produced per reproduction event")
    aparser.add_argument("--MFP-Dm", default='1',
                         help="Development model: 0=uniform, 1=thermal accumulation")
    aparser.add_argument("--MFP-TuSD", default='0.05',
                         help="Variation in thermal unit transition")
    aparser.add_argument("--MFP-Tmax", default='35',
                         help="Maximum temperature under which development can occur")
    aparser.add_argument("--MFP-nR", default='2500',
                         help="Number of simulations per run")
    aparser.add_argument("--MFP-nT", default='20',
                         help="Number of threads to utilize")
    aparser.add_argument("--MFP-Mx", default='500000',
                         help="Maximum number of flies allowed")
    aparser.add_argument("--MFP-seed", default='1',
                         help="Random number generator seed")
    # Set per-runset in run script or fixed...
    # aparser.add_argument("--MFP-T", default='dummy_value', help="")
    # aparser.add_argument("--MFP-o", default='dummy_value', help="")
    # aparser.add_argument("--MFP-q", default='true', help="")
    # aparser.add_argument("--MFP-pr", default='false', help="")
    # aparser.add_argument("--MFP-plot", default='false', help="")

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

    Fss = pd.to_datetime(args.SS_start)

    current_year = Fss.year
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

    #######################################
    ## setup or update medfoes runs
    # vars: @ TCC for making this a function
    # tempdf
    local_time_offset = args.local_time_offset
    medfoes_dir = os.path.join(args.basedir,args.medfoes_dir)
    medfoes_runs_per_date = args.MFP_nR
    start_date=Fss
    start_year=START_YEAR
    end_year=END_YEAR
    medfoes_num_years_of_temp_data_per_set=args.medfoes_num_years_of_temp_data_per_set
    force_full_medfoes_rerun=args.force

    datestr = '{}-'+'{:02d}-{:02d}'.format(Fss.month, Fss.day)
    year_range = np.arange(start_year, end_year+1)
    if year_range[-1] < current_year: # include current year
        year_range = np.append(year_range, current_year)

    os.makedirs(os.path.join(medfoes_dir,'temps'), exist_ok=True)
    out_fns = []
    all_out_fns = []
    logging.info("Checking MEDFOES temperature input files for changes")
    for y in year_range:
        t = tempdf.loc[datestr.format(y):datestr.format(y+medfoes_num_years_of_temp_data_per_set)]
        out_fn = "temps_{:04d}-{:02d}-{:02d}.csv".format(t.index[0].year, t.index[0].month, t.index[0].day)
        out_fullfn = os.path.join(medfoes_dir, 'temps', out_fn)
        all_out_fns.append(out_fn)
        # potentially skip if input temperature file is not changed
        if not force_full_medfoes_rerun:
            try:
                tmp = temperature_functions.load_temperature_csv(out_fullfn, local_time_offset=local_time_offset)
            except FileNotFoundError:
                tmp = None
            if( tmp is not None and
                    t.shape == tmp.shape and
                    all(np.isclose(t.values, tmp.values))
                    and all(t.index == tmp.index) ):
                logging.info("{} no change".format(out_fn))
                continue
            else:
                logging.info("{} CHANGED".format(out_fn))
        # output new temperature file
        logging.info("Updating {}".format(out_fn))
        out_fns.append(out_fn)
        t.to_csv(out_fullfn, columns=['AT'])

    ## Write Medfoes-p configuration mfp.cfg
    with open(os.path.join(medfoes_dir, "mfp.cfg"), 'w') as fh:
        for action in aparser._actions:
            if action.dest.startswith('MFP_'):
                print(action.dest[4:], action.default, file=fh)  # could also output action.help
        print('T', 'dummy_value_overridden_by_command_line_argument', file=fh)
        print('o', 'dummy_value_overridden_by_command_line_argument', file=fh)
        print('q', 'true', file=fh)
        print('pr', 'false', file=fh)
        print('plot', 'false', file=fh)

    # ## Output the list of temperature files used by run scripts
    # with open(os.path.join(medfoes_dir, "temps_list"),'w') as fh:
    #     print('\n'.join([os.path.join('temps',x) for x in out_fns]), file=fh)

        #     ## output sge run file
    sge_runfile_pat = """#!/bin/bash
#$-N med-foes-p_longrun
#$-S /bin/bash
#$-o sge_$JOB_ID.out
#$-e sge_$JOB_ID.err
#$-q all.q
#$-pe orte 15
#$-cwd
#$-V
#$-m n
#$-sync yes
#$-b yes
#$-p -64
#$-t 1-{NUM_RUNSETS:d}

BASEDIR="$PWD"
TFILES=({TFILES})

TFILE=${{TFILES[$((${{SGE_TASK_ID}}-1))]}}
tmp="${{TFILE##*/temps/temps_}}"
datestamp="${{tmp%.csv}}"
OUTDIR="runs/$datestamp"

JAR="$BASEDIR/MedFoesP-0.6.2.jar"
CFGFILE="$BASEDIR/mfp.cfg"
JAVA='/home/travc/jdk/jdk1.8.0_131/bin/java'
NICE_LVL=9

echo '#####' Running $SGE_TASK_ID out=$OUTDIR
rm -rf "$OUTDIR" && \\
mkdir -p "$OUTDIR" && \\
nice -n $NICE_LVL \\
$JAVA -jar "$JAR" -f "$CFGFILE" -nR {NUM_RUNS_PER_SET} -T "$TFILE" -o "$OUTDIR" && \\
tar --remove-files -C "$OUTDIR" -cjf "$OUTDIR/Runs.tar.bz2" Runs
"""

    # sge run file for all years
    with open(os.path.join(medfoes_dir, "run_mfp.sge"),'w') as fh:
        print(sge_runfile_pat.format(
            MEDFOES_DIR=medfoes_dir,
            NUM_RUNSETS=len(out_fns),
            NUM_RUNS_PER_SET=medfoes_runs_per_date,
            TFILES='\n'.join(['"'+os.path.join("${BASEDIR}",'temps',x)+'"' for x in all_out_fns])),
            file=fh)
    os.chmod(os.path.join(medfoes_dir, "run_mfp.sge"), 0o774)
#    # sge run file for just doing updated
#    if len(out_fns) > 0:
#        with open(os.path.join(medfoes_dir, "run_mfp_update.sge"),'w') as fh:
#            print(sge_runfile_pat.format(
#                MEDFOES_DIR=medfoes_dir,
#                NUM_RUNSETS=len(out_fns),
#                NUM_RUNS_PER_SET=medfoes_runs_per_date,
#                TFILES='\n'.join(['"'+os.path.join("${BASEDIR}",'temps',x)+'"' for x in out_fns])),
#                file=fh)
#        os.chmod(os.path.join(medfoes_dir, "run_mfp_update.sge"), 0o774)
#    else:
#        logging.info("No temperature files changed, so no medfoes sets to update")
#        #os.unlink(os.path.join(medfoes_dir, "run_mfp_update.sge"))

    ## output current run file
    current_runfile_pat = """#!/bin/bash
BASEDIR="$PWD"
TFILE="{CURRENT_TFILE}"
tmp="${{TFILE##*/temps/temps_}}"
datestamp="${{tmp%.csv}}"
OUTDIR="runs/$datestamp"

JAR="$BASEDIR/MedFoesP-0.6.2.jar"
CFGFILE="$BASEDIR/mfp.cfg"
JAVA='/home/travc/jdk/jdk1.8.0_131/bin/java'
NICE_LVL=9

echo '#####' Running $SGE_TASK_ID out=$OUTDIR
rm -rf "$OUTDIR" && \\
mkdir -p "$OUTDIR" && \\
nice -n $NICE_LVL \\
$JAVA -jar "$JAR" -f "$CFGFILE" -nR {NUM_RUNS_PER_SET} -T "$TFILE" -o "$OUTDIR" && \\
tar --remove-files -C "$OUTDIR" -cjf "$OUTDIR/Runs.tar.bz2" Runs
"""

    if len(out_fns) > 0:
        with open(os.path.join(medfoes_dir, "run_mfp_current.sh"), 'w') as fh:
            print(current_runfile_pat.format(
                MEDFOES_DIR=medfoes_dir,
                NUM_RUNS_PER_SET=medfoes_runs_per_date,
                CURRENT_TFILE=os.path.join("${BASEDIR}", 'temps', out_fns[-1])),
                file=fh)
        os.chmod(os.path.join(medfoes_dir, "run_mfp_current.sh"), 0o774)

    ## end
    logging.info("Done: {:.2f} sec elapsed".format(time.time()-tic_total))
    return 0




#########################################################################
# Main loop hook... if run as script run main, else this is just a module
if __name__ == "__main__":
    sys.exit(Main(argv=None))
