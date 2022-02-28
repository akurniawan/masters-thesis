for FILE in run_*; do 
    qsub $FILE;
done;

# qsub