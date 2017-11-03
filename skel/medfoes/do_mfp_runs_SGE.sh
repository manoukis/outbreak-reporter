#!/bin/bash

#EMAIL=

cmd="qsub -N \"MedFoesP-array\" \"$PWD/run_mfp.sge\""

if [[ -z "${EMAIL}" ]]; then
  echo "WARNING: No EMAIL variable set, will not send notification"
fi

START_DATE=`date -Ins`
echo "######## STARTED ########"
echo "## $START_DATE "

bash -c "$cmd"
rval=$?

END_DATE=`date -Ins`
echo '######## FISNISHED ########'
echo "## $END_DATE "
echo $cmd
echo '###########################'

if [[ ! -z "${EMAIL}" ]]; then
echo "Job Finished:
####
# $cmd
####
rval  : $rval
start : $START_DATE
end   : $END_DATE" | \
mail -s "Job Finsihed: $cmd" $EMAIL
fi

