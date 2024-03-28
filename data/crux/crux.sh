#!/bin/sh

# Edit version - type of classification here
crux_version=202401
model_version=chrome4
classification_type=topics-api

# Variables
prefix=crux_${crux_version}
csv_path=${prefix}.csv
domains_path=${prefix}.domains
classified_path=${prefix}_${model_version}_${classification_type}.tsv

# Download CrUX top list
if [ ! -f $csv_path ]
then
    wget -q -O $csv_path.gz https://github.com/zakird/crux-top-lists/raw/main/data/global/$crux_version.csv.gz
    gzip -cdk $csv_path.gz > $csv_path
    rm $csv_path.gz
fi

#check if domains extracted
if [ ! -f $domains_path ]
then
    sed -nr "s/https?:\/\/(.*),.*/\1/p" $csv_path > $domains_path.tmp
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
