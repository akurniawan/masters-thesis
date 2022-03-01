for FILE in run_*; do 
    echo $FILE;
    qsub $FILE;
done;

# qsub