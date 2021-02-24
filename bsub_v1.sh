#!/bin/bash
#source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh #this sets environment for psana2. note psana2 not compatible with psana(1)
base_path=/reg/neh/home/tdd14/TMO/scriptslw07
script=$base_path/preproc_v1.py
log=/reg/d/psdm/tmo/tmoc00118/scratch/preproc/logs/run$1_v1.log

if [ -z "$2" ]
then
    nodes=16
else
    nodes=$2
fi

echo $log
echo $script

export PS_SRV_NODES=1
bsub -o $log -q psfehq -n $nodes mpirun python $script $1
