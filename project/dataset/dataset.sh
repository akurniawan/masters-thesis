declare -a dataset=("CCMatrix/v1" "WikiMatrix/v1" "wikimedia/v2021402" "CCAligned/v1" "Tanzil/v1" "XLEnt/v1.1" "OpenSubtitles/v2018" "QED/v2.0a" "TED2020/v1" "GNOME/v1" "bible-uedin/v1" "News-Commentary/v16" "GlobalVoices/v2018q4" "KDE4/v2" "tico-19/v2020-10-28" "Tatoeba/v2021-07-22" "ELRC_2922/v1" "Ubuntu/v14.10")

mkdir -p {moses,mono}

## now loop through the above array
# for i in "${dataset[@]}"
# do
#     echo $i
#     arr_dataset=(${i//// })
#     # wget "https://object.pouta.csc.fi/OPUS-$i/moses/en-id.txt.zip"
#     wget -c "https://object.pouta.csc.fi/OPUS-$i/mono/en.tok.gz" -P "mono/${arr_dataset[0]}"
#     wget -c "https://object.pouta.csc.fi/OPUS-$i/mono/id.tok.gz" -P "mono/${arr_dataset[0]}"
# done

# declare -a mono_dataset=("JW300/v1c" "Mozilla-I10n/v1")
# for i in "${mono_dataset[@]}"
# do
#     arr_dataset=(${i//// })
#     wget -c "https://object.pouta.csc.fi/OPUS-$i/mono/en.tok.gz" -P "mono/${arr_dataset[0]}"
#     wget -c "https://object.pouta.csc.fi/OPUS-$i/mono/id.tok.gz" -P "mono/${arr_dataset[0]}"
# done

# unzip tmp/\*.zip -d tmp_ext/
mv tmp_ext/*.id moses/
mv tmp_ext/*.en moses/
rm -rf tmp_ext