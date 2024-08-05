import argparse
import os
import numpy as np
import pandas as pd
import json

import simulator_library


class User:
    def __init__(self, panelist_id, id, epoch_topics):
        self.panelist_id = panelist_id
        self.id = id
        self.epoch_topics = epoch_topics
        # third party A
        self.a_topics_view = None  # 1D array
        self.a_ground_truth = None  # 1D array
        self.a_genuine_topics = None  # Matrix
        self.a_noisy_topics = None  # Matrix
        self.a_observed_topics = None  # Matrix
        # third party B
        self.b_topics_view = None  # 1D array
        self.b_ground_truth = None  # 1D array
        self.b_genuine_topics = None  # Matrix
        self.b_noisy_topics = None  # Matrix
        self.b_observed_topics = None  # Matrix

    def generate_topics_view(self, nb_epochs, taxonomy_ids, p=0.05):
        topics_view = []
        topics_ground_truth = []
        for epoch in range(nb_epochs):
            topic, gt = simulator_library.topics_call(
                self.epoch_topics[epoch], taxonomy_ids, p
            )
            topics_view.append(topic)
            topics_ground_truth.append(gt)
        return topics_view, topics_ground_truth

    def init_experiments_a(self, nb_epochs, taxonomy_ids, unobserved_topics, p=0.05):
        if self.a_topics_view == None:
            self.a_topics_view, self.a_ground_truth = self.generate_topics_view(
                nb_epochs, taxonomy_ids, p
            )
        if self.a_genuine_topics == None:
            (
                self.a_genuine_topics,
                self.a_noisy_topics,
                self.a_observed_topics,
            ) = simulator_library.denoise(self.a_topics_view, unobserved_topics)
        return (
            self.id,
            self.a_topics_view,
            self.a_ground_truth,
            self.a_genuine_topics,
            self.a_noisy_topics,
            self.a_observed_topics,
        )

    def init_experiments_b(self, nb_epochs, taxonomy_ids, unobserved_topics, p=0.05):
        if self.b_topics_view == None:
            self.b_topics_view, self.b_ground_truth = self.generate_topics_view(
                nb_epochs, taxonomy_ids, p
            )
        if self.b_genuine_topics == None:
            (
                self.b_genuine_topics,
                self.b_noisy_topics,
                self.b_observed_topics,
            ) = simulator_library.denoise(self.b_topics_view, unobserved_topics)
        return (
            self.id,
            self.b_topics_view,
            self.b_ground_truth,
            self.b_genuine_topics,
            self.b_noisy_topics,
            self.b_observed_topics,
        )


#################


def unobserved_topics_ids(df_top_list, threshold, taxonomy_ids):
    df_topics = (
        df_top_list.groupby("topic")["domain"].nunique().to_frame().reset_index()
    )
    df = df_topics[df_topics["domain"] > threshold]
    genuine_topics = set(list(df["topic"].values))
    all_topics = set(taxonomy_ids)
    unobserved_topics = all_topics.difference(genuine_topics)
    return np.array(list(unobserved_topics))


def create_users(df_users_topics, nb_epochs, taxonomy_ids, T, repeat_each_user_n_times):
    users = []
    id = 0
    for panelist_id in df_users_topics["panelist_id"].unique():
        df_user = df_users_topics[df_users_topics["panelist_id"] == panelist_id]
        topT_epochs = []
        for epoch in range(nb_epochs):
            topT = df_user[df_user["epoch_id"] == epoch]["topic"].tolist()

            # if topT is not size T, the spec says that we pad with random
            # topics from the taxonomy. However, and because of the witness
            # requirement, these random topics will never get returned to third
            # parties that have not observed them.

            # Thus commenting out following lines of code:
            # if len(topT) != T:
            # check if correct size, if not draw randomly from taxonomy
            #     possible_choices = list(set(taxonomy_ids).difference(topT))
            #     for topic in np.random.choice(
            #         possible_choices, T - len(topT), replace=False
            #     ):
            #         topT.append(topic)


            # append to matrix, order topics according to id
            topT_epochs.append(np.sort(topT))
        for _ in range(repeat_each_user_n_times):
            users.append(User(panelist_id, id, topT_epochs))
            id += 1
    return users


#################

if __name__ == "__main__":
    # Create Argument Parser
    parser = argparse.ArgumentParser(
        prog="python3 topics_simulator.py",
        description="Simulate the Topics API and evaluate its privacy guarantees",
    )
    parser.add_argument("users_topics_tsv")
    parser.add_argument("nb_epochs", type=int)
    parser.add_argument("config_model_json")
    parser.add_argument("top_list_tsv")
    parser.add_argument("unobserved_topics_threshold", type=int)
    parser.add_argument("repeat_each_user_n_times", type=int)
    parser.add_argument("output_prefix")
    args = parser.parse_args()

    if (
        not (os.path.isfile(args.users_topics_tsv))
        or not (os.path.isfile(args.config_model_json))
        or not (os.path.isfile(args.top_list_tsv))
    ):
        raise Exception("Error: file(s) missing")
    else:
        # load config.json
        with open(args.config_model_json, "r") as f:
            config = json.load(f)

        nb_epochs = args.nb_epochs
        df_users_topics = pd.read_csv(args.users_topics_tsv, sep="\t")

        model_dirname = os.path.dirname(args.config_model_json)

        taxonomy = pd.read_csv(
            model_dirname + "/" + config["taxonomy_filename"], sep="\t"
        )
        taxonomy_ids = taxonomy[config["taxonomy_id_column"]].unique()

        df_top_list = pd.read_csv(args.top_list_tsv, sep="\t")
        df_top_list.drop(
            df_top_list[df_top_list["topic"] == config["unknown_topic_id"]].index,
            inplace=True,
        )
        unobserved_topics = unobserved_topics_ids(
            df_top_list, args.unobserved_topics_threshold, taxonomy_ids
        )

        users = create_users(
            df_users_topics,
            nb_epochs,
            taxonomy_ids,
            config["max_categories"],
            args.repeat_each_user_n_times,
        )

        # Init experiments
        simulator_library.init_exp_a(
            users, unobserved_topics, taxonomy_ids, nb_epochs, 0.05
        )

        simulator_library.init_exp_b(
            users, unobserved_topics, taxonomy_ids, nb_epochs, 0.05
        )

        # Denoise from third party perspective
        simulator_library.denoise_exp_all_epochs_a(users, nb_epochs, args.output_prefix)
        simulator_library.denoise_exp_all_epochs_b(users, nb_epochs, args.output_prefix)

        # Reidentification experiment
        simulator_library.reidentification_exp_all_epochs(
            users, nb_epochs, taxonomy_ids, args.output_prefix
        )
