#!/bin/sh

# Edit version - type of classification here
tranco_version=G6KQK
model_version=chrome5
classification_type=topics-api

# Variables
prefix=tranco_${tranco_version}
csv_path=${prefix}.csv
domains_path=${prefix}.domains
classified_path=${prefix}_${model_version}_${classification_type}.tsv

# Download Tranco top list
if [ ! -f $csv_path ]
then
    wget -q -O $csv_path https://tranco-list.eu/download/$tranco_version/1000000
fi

#check if domains extracted
if [ ! -f $domains_path ]
then
    sed -nr "s/[0-9]+,(.*)/\1/p" $csv_path > $domains_path.tmp
    sed -i 's/\r$//g' $domains_path.tmp
    sort $domains_path.tmp | uniq > $domains_path
    rm $domains_path.tmp
fi

if [ ! -f $classified_path ]
then
    #Header
    echo "domain\ttopic" > $classified_path
    #Parallel inference
    parallel -X --bar -N 1000 -a $domains_path -I @@ "python3 ../../topics_classifier/classify.py -mv $model_version -ct $classification_type -i @@ >> $classified_path"
fi
