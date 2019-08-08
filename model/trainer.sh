read -p "desc: " desc

dt=$(date +"D%d_%T")
logdir=logs
logfile=${logdir}/${dt}_${desc}.log

touch ${logfile}
python trainer.py --epochs 100 --logfile ${logfile} &>/dev/null &

multitail -n 1000 ${logfile}
