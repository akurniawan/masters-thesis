for FILE in run_*; do 
    echo $FILE;
    if grep -q "zsbert" <<< "$FILE"; then
        qsub $FILE;
    fi
    # qsub $FILE;
done;