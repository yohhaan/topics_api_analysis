import argparse
import os
import pandas as pd
import re
import json
from datetime import datetime


def clean_browsing_df(df_browsing):
    df_browsing.drop("web_visits_id", axis=1, inplace=True)
    df_browsing.drop("id", axis=1, inplace=True)
    df_browsing.drop("url", axis=1, inplace=True)
    df_browsing.drop("active_seconds", axis=1, inplace=True)
    df_browsing["FQDN"] = df_browsing["subdomain"].fillna("") + df_browsing["domain"]
    df_browsing.drop("domain", axis=1, inplace=True)
    df_browsing.drop("subdomain", axis=1, inplace=True)
    df_browsing.rename(columns={"FQDN": "domain"}, inplace=True)
    df_browsing["used_at"] = pd.to_datetime(
        df_browsing["used_at"], format="%Y-%m-%d %H:%M:%S"
    )


def return_epoch_id(epoch_cutoff, time):
    if time < epoch_cutoff:
        return i


if __name__ == "__main__":
    # Create Argument Parser
    parser = argparse.ArgumentParser(
        prog="python3 extract_domains.py",
        description="Extract domains from web raw data.",
    )
    parser.add_argument("web_data_tsv")
    parser.add_argument("browsing_csv")
    parser.add_argument("epochs_json")
    parser.add_argument("config_model_json")
    parser.add_argument("output_tsv")
    args = parser.parse_args()

    if (
        not (os.path.isfile(args.web_data_tsv))
        or not (os.path.isfile(args.browsing_csv))
        or not (os.path.isfile(args.epochs_json))
        or not (os.path.isfile(args.config_model_json))
    ):
        raise Exception("Error: file(s) missing")
    else:
        # load config.json
        with open(args.config_model_json, "r") as f:
            config = json.load(f)

        # load epochs.json
        with open(args.epochs_json, "r") as f:
            epochs = json.load(f)

        model_dirname = os.path.dirname(args.config_model_json)

        # parse utility buckets
        utility_buckets = pd.read_csv(
            model_dirname + "/" + config["utility_buckets_filename"], sep="\t"
        )
        for level in config["utility_buckets_utility_levels"]:
            utility_buckets[config["utility_buckets_utility_column"]].replace(
                level["name"], level["level"], inplace=True
            )

        # Raw browsing data and Topics
        df_browsing = pd.read_csv(args.browsing_csv, sep=",")
        web_data = pd.read_csv(args.web_data_tsv, sep="\t")
        taxonomy = pd.read_csv(
            model_dirname + "/" + config["taxonomy_filename"], sep="\t"
        )
        clean_browsing_df(df_browsing)

        # extract data for each epoch, we use a for loop for epochs that overlap
        df_epochs = pd.DataFrame()

        for i in range(len(epochs["epochs_cut_off"])):
            previous_epoch_cut_off = datetime.strptime(
                epochs["epochs_cut_off"][i][0], "%Y-%m-%d %H:%M:%S"
            )
            current_epoch_cut_off = datetime.strptime(
                epochs["epochs_cut_off"][i][1], "%Y-%m-%d %H:%M:%S"
            )

            df = df_browsing[
                (df_browsing["used_at"] >= previous_epoch_cut_off)
                & (df_browsing["used_at"] < current_epoch_cut_off)
            ]
            df["epoch_id"] = i
            df_epochs = pd.concat([df_epochs, df], ignore_index=True)

        # now merge with classification data
        merged = df_epochs.merge(web_data, on="domain", how="inner")
        df_topics = merged.groupby(
            ["panelist_id", "epoch_id", "topic"], as_index=False
        ).agg(count=("topic", "count"))

        # drop unknown topics
        df_users = df_topics[df_topics["topic"] != config["unknown_topic_id"]]

        # parse taxonomy to match utility buckets
        taxonomy[config["utility_buckets_toplevel_column"]] = taxonomy[
            config["taxonomy_name_column"]
        ].apply(lambda x: re.findall("(\/.*?)\/", x + "/")[0])
        utility = taxonomy.merge(
            utility_buckets, on=config["utility_buckets_toplevel_column"], how="inner"
        )
        utility.drop(config["taxonomy_name_column"], axis=1, inplace=True)
        utility.drop(config["utility_buckets_toplevel_column"], axis=1, inplace=True)
        utility.rename(columns={config["taxonomy_id_column"]: "topic"}, inplace=True)
        utility.rename(
            columns={config["utility_buckets_utility_column"]: "utility"}, inplace=True
        )

        # Add utility metric to users profiles
        df_users_utility = df_users.merge(utility, on="topic", how="inner")

        # sort according to utility bucket and frequency, keep only max categories
        df_final = (
            df_users_utility.sort_values(
                by=["utility", "count"], ascending=[False, False]
            )
            .groupby(["panelist_id", "epoch_id"])
            .head(config["max_categories"])
        )
        df_final.drop("utility", axis=1, inplace=True)
        df_final.drop("count", axis=1, inplace=True)

        # filter to only users with topics all epochs
        df_nb_epochs = df_final.groupby(["panelist_id"], as_index=False)[
            "epoch_id"
        ].agg(["nunique"])

        df = df_final.merge(df_nb_epochs, on="panelist_id", how="inner")
        df_users_filtered = df.drop(
            df[df["nunique"] < len(epochs["epochs_cut_off"])].index
        )
        df_users_filtered.drop("nunique", axis=1, inplace=True)

        # save to tsv
        df_users_filtered.to_csv(args.output_tsv, sep="\t", index=False)
