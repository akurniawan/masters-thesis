for FILE in run_adapter_*; do 
    echo $FILE;
    qsub $FILE;
    # qsub $FILE;
done;