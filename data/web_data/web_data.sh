#!/bin/sh

# Edit version - type of classification here
model_version=chrome4
classification_type=topics-api
nb_epochs=5

# Variables
prefix=web_data
csv_path=browsing.csv
domains_path=${prefix}.domains
classified_path=${prefix}_${model_version}_${classification_type}.tsv
epochs_json=epochs_${nb_epochs}_weeks.json
users_topics_epochs=users_topics_${nb_epochs}_weeks.tsv

# Download Web Data - Can take a moment
if [ ! -f $csv_path ]
then
    web_tracking_data=web_tracking_data.tar.gz
    wget -O $web_tracking_data https://zenodo.org/records/4757574/files/web_tracking_data.tar.gz?download=1
    tar -xvzf $web_tracking_data web_routineness_release/raw/browsing.csv --strip-components=2
    rm $web_tracking_data
fi

#check if domains extracted
if [ ! -f $domains_path ]
then
    python3 extract_domains.py $csv_path $domains_path
fi

if [ ! -f $classified_path ]
then
    #Header
    echo "domain\ttopic" > $classified_path
    #Parallel inference
    parallel -X --bar -N 1000 -a $domains_path -I @@ "python3 ../../topics_classifier/classify.py -mv $model_version -ct $classification_type -i @@ >> $classified_path"
fi

#users profiles not generated
if [ ! -f $users_topics_epochs ]
then
    python3 create_topics_profiles.py $classified_path $csv_path $epochs_json ../../topics_classifier/$model_version/config.json $users_topics_epochs
fi
