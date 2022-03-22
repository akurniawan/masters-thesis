for FILE in run_*; do 
    echo $FILE;
    if [[ $FILE =~ "bigadapter" || $FILE =~ "zbert" ]]; then
        qsub $FILE;
    fi
    # qsub $FILE;
done;