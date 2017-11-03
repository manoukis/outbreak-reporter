Thu Nov  2 23:48:47 HST 2017

Current workflow:

`cp -r skel <short_name>`  
`cd <short_name>`  
`mkdir official`  
copy any SITSTAT or lifecyle docs to official directory  
edit main.cfg  
`../bin/fetch_and_process_temperatures.py`  
`../bin/setup_medfoes.py`  
`cd medfoes`  
`do_mfp_runs_SGE.sh`  
wait for medfoes runs to complete  
`cd ..`  
`../bin/make_report.py`  
report.pdf is the final output  
