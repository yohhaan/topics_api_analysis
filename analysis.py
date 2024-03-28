from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns


def savefig(path, size=[4, 3]):
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["figure.autolayout"] = True
    # Sane default fig size for papers
    matplotlib.rcParams["figure.figsize"] = [4, 3]

    # Uses Opentype-compatible fonts
    # conferences often require this for camera ready, so if you don't do it pre-submission you'll have a nightmare at camera-ready time.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # Automatically make the directory hierarchy so I can just save figures with path names
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Makes background transparent so plots can go in slides and look good
    plt.gcf().patch.set_alpha(0)
    # Default fig size
    plt.gcf().set_size_inches(*size)
    # Make figure fill whole PDF (otherwise figs have huge margins in LaTeX
    plt.tight_layout(
        pad=0,
    )
    plt.savefig(path, bbox_inches="tight")
    plt.clf()
    # Sets seaborn whitegrid on every plot for consistency (darkgrid is nice for slides)
    sns.set_style("whitegrid")


def cdf_domains_per_topic(df, taxonomy, output_path):
    """
    Histplot of topics count binned per number of domain(s) for each topic
    Ignore Unknown topic
    """
    # add topics potentially non observed / note: -2 is discarded
    merged = df.merge(pd.DataFrame({"topic": taxonomy["ID"]}), how="right", on="topic")
    # get number of domains for each topic
    data = merged.groupby("topic")["domain"].nunique().reset_index()
    plt.clf()
    # ecdf
    plot = sns.ecdfplot(data=data, x="domain", stat="count")
    plot.set(
        xlabel="Number of unique domain(s) observations", ylabel="Number of topics"
    )
    plot.set_xscale("symlog")
    plot.set_xlim(left=0)
    plot.set_ylim([0, 500])
    savefig(output_path + "_cdf_domains_per_topic.pdf")

    with open(output_path + "_domains_per_topics.stats", "w") as f:
        f.write("Stats about topics per domain:\n {} \n".format(data.describe()))
        for t in [0, 1, 5, 10, 50, 100, 200, 300, 400]:
            f.write(
                "Nb topics observed on <= {} domains: {}\n".format(
                    t, data[data["domain"] <= t]["topic"].nunique()
                )
            )


def extract_stats(df, nb_epochs, df_browsing=None):
    N = df["panelist_id"].nunique()

    print("nb users:", N)

    if df_browsing is not None:
        mask = df_browsing["panelist_id"].isin(df["panelist_id"].unique())
        urls_filtered = df_browsing[mask]
        print("nb URLS visited:", len(urls_filtered))
        urls_filtered["FQDN"] = (
            urls_filtered["subdomain"].fillna("") + urls_filtered["domain"]
        )
        print("nb unique domains visited:", urls_filtered["domain"].nunique())
        print("nb unique FQDNS visited:", urls_filtered["FQDN"].nunique())

    print("Unique topics &", end="")
    for nb in df.groupby("epoch_id")["topic"].nunique():
        print("{} &".format(nb), end="")
    print("")

    print("Unique profiles &", end="")
    for epoch in range(nb_epochs):
        profiles = {}
        for id in df["panelist_id"].unique():
            topT = np.sort(
                df[(df["panelist_id"] == id) & (df["epoch_id"] == epoch)][
                    "topic"
                ].to_numpy()
            )
            profile = ""
            for topic in topT:
                profile += str(topic) + "-"
            profiles[profile] = 0

        print(
            "{} &".format(len(profiles.keys())),
            end="",
        )
    print("")

    for epoch in range(nb_epochs - 1):

        nb_stable_topics = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for id in df["panelist_id"].unique():
            df_user = df[df["panelist_id"] == id]
            epoch_a = df_user[df_user["epoch_id"] == epoch]["topic"]
            epoch_b = df_user[df_user["epoch_id"] == epoch + 1]["topic"]

            intersection = set(epoch_a).intersection(epoch_b)

            nb_stable_topics[len(intersection)] += 1

        print("From week {} to {} &".format(epoch + 1, epoch + 2), end="")
        for nb in nb_stable_topics.values():
            print("{} ({}\%) &".format(nb, round(nb / N * 100, 1)), end="")
        print("")


def plot_cdf_size_reidentified_groups(input_prefix, output_path, nb_epochs, nb_users):
    df = pd.DataFrame(columns=["k", "Epochs"])

    for epoch in range(nb_epochs):
        filename = (
            input_prefix + "_epoch_" + str(epoch) + "_size_reidentified_groups.npy"
        )

        size_epoch = np.load(filename)
        df = pd.concat(
            [df, pd.DataFrame({"k": size_epoch, "Epochs": epoch + 1})],
            ignore_index=True,
        )
    df["k"] = df["k"].replace(0, nb_users)
    plt.clf()
    plot = sns.ecdfplot(
        data=df[df["Epochs"] >= 3],
        x="k",
        stat="proportion",
        hue="Epochs",
    )
    sns.move_legend(
        plot,
        "upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        title="Weeks",
        frameon=False,
        reverse=True,
    )
    plot.set(xlabel="$k$", ylabel="Proportion of users")
    plot.set_xscale("symlog")
    plot.set_xlim([0.8, nb_users * 1.2])
    savefig(output_path + "_cdf_size_groups.pdf")


# # PLOTS
def plot_multi_shot_denoise(input_prefix, output_path, nb_epochs_total):
    epochs = [i + 1 for i in range(nb_epochs_total)]

    accuracy = np.load(input_prefix + "_accuracy.npy")
    precision = np.load(input_prefix + "_precision.npy")
    tp_rate = np.load(input_prefix + "_tpr.npy")
    fp_rate = np.load(input_prefix + "_fpr.npy")

    data = pd.DataFrame(
        {
            "Epochs": epochs,
            "Accuracy": accuracy,
            "Precision": precision,
            "TPR": tp_rate,
            "FPR": fp_rate,
        }
    )
    plt.clf()
    ax = sns.lineplot(
        data=data[data["Epochs"] >= 3],
        x="Epochs",
        y="Accuracy",
        marker="o",
        color="#005AB5",
        linewidth=3,
        label="Accuracy",
    )
    ax.set_ylabel("Accuracy", color="#005AB5")
    # ax.lines[0].set_linestyle("--")

    ax2 = ax.twinx()
    sns.lineplot(
        data=data[data["Epochs"] >= 3],
        x="Epochs",
        y="Precision",
        ax=ax2,
        marker="P",
        color="#D41159",
        label="Precision",
    )
    ax2.set_ylabel("Precision", color="#D41159")
    ax2.grid(None)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(h1 + h2, l1 + l2)
    ax2.get_legend().remove()
    savefig(output_path + "_denoise_accuracy_precision.pdf")

    plt.clf()
    ax = sns.lineplot(
        data=data[data["Epochs"] >= 3],
        x="Epochs",
        y="FPR",
        marker="o",
        color="#1AFF1A",
        label="FPR",
    )

    ax.set_ylabel("FPR", color="#1AFF1A")

    ax2 = ax.twinx()
    sns.lineplot(
        data=data[data["Epochs"] >= 3],
        x="Epochs",
        y="TPR",
        ax=ax2,
        marker="P",
        color="#4B0092",
        label="TPR",
    )
    ax2.set_ylabel("TPR", color="#4B0092")
    ax2.grid(None)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.get_legend().remove()
    ax2.get_legend().remove()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    savefig(output_path + "_denoise_tpr_fpr.pdf")


def load_override_list(override_list_path: str) -> None:
    # Load manually curated list
    df = pd.read_csv(override_list_path, sep="\t")
    domain_column = []
    topic_column = []
    for _, row in df.iterrows():
        topics = row["topics"]
        # check if topics column is empty
        if type(topics) is not str and math.isnan(topics):
            domain_column.append(row["domain"])
            topic_column.append(-2)
        else:
            for topic in topics.split(","):
                domain_column.append(row["domain"])
                topic_column.append(int(topic))
    return pd.DataFrame({"domain": domain_column, "topic": topic_column})


if __name__ == "__main__":
    taxonomy = pd.read_csv("topics_classifier/chrome4/taxonomy.tsv", sep="\t")

    ## Distributions
    crux = pd.read_csv("data/crux/crux_202401_chrome4_topics-api.tsv", sep="\t")
    cdf_domains_per_topic(crux, taxonomy, "data/figs/crux")
    tranco = pd.read_csv("data/tranco/tranco_6JZJX_chrome4_topics-api.tsv", sep="\t")
    cdf_domains_per_topic(tranco, taxonomy, "data/figs/tranco")
    web_data = pd.read_csv("data/web_data/web_data_chrome4_topics-api.tsv", sep="\t")
    cdf_domains_per_topic(web_data, taxonomy, "data/figs/web_data")
    override = load_override_list("topics_classifier/chrome4/override_list.tsv")
    cdf_domains_per_topic(override, taxonomy, "data/figs/override")

    ## Stats
    nb_epochs = 5
    df_browsing = pd.read_csv("data/web_data/browsing.csv", sep=",")

    df_users_topics = pd.read_csv(
        "data/web_data/users_topics_" + str(nb_epochs) + "_weeks.tsv", sep="\t"
    )
    extract_stats(df_users_topics, nb_epochs, df_browsing)

    ## Reidentificaiton experiment
    nb_epochs = 5
    plot_cdf_size_reidentified_groups(
        "data/reidentification_exp/" + str(nb_epochs) + "_weeks_10_unobserved",
        "data/figs/" + str(nb_epochs) + "_weeks_10_unobserved",
        nb_epochs,
        1207,
    )

    ## Denoise experiment
    nb_repetitions = 100
    nb_epochs = 5

    plot_multi_shot_denoise(
        "data/denoise_exp/"
        + str(nb_epochs)
        + "_weeks_"
        + str(nb_repetitions)
        + "_repetitions_10_unobserved_denoise_a",
        "data/figs/" + str(nb_epochs) + "_weeks_" + str(nb_repetitions) + "_10_a",
        5,
    )
    plot_multi_shot_denoise(
        "data/denoise_exp/"
        + str(nb_epochs)
        + "_weeks_"
        + str(nb_repetitions)
        + "_repetitions_10_unobserved_denoise_b",
        "data/figs/" + str(nb_epochs) + "_weeks_" + str(nb_repetitions) + "_10_b",
        5,
    )
