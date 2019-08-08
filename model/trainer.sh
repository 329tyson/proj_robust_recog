read -p "desc: " desc

dt=$(date +"D%d_%T")
logdir=logs
logfile=${logdir}/${dt}_${desc}.log

touch ${logfile}
python trainer.py --epochs 1 --logfile ${logfile} &>/dev/null &

multitail ${logfile}
