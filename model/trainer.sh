case $1 in
    "--f")
        echo "PROCESS IS RUNNING FOREGROUND"
        front=true
        ;;
    "-f")
        echo "PROCESS IS RUNNING FOREGROUND"
        front=true
        ;;
    *)
        front=false
        ;;
esac

dt=$(date +"D%d_%T")
logdir=logs
logfile=${logdir}/${dt}_${desc}.log
touch ${logfile}

cmd="python trainer.py --logfile ${logfile}"

if [ ${front} ]; then
    ${cmd}
    rm ${logfile}
    exit
fi

read -p "desc: " desc
cmd &>/dev/null &


multitail -n 1000 ${logfile}
