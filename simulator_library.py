import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import repeat
import os

# force reseed (https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing)
os.register_at_fork(after_in_child=np.random.seed)

def topics_call(top_topics, taxonomy_ids, p=0.05):
    """Simulate a call to TOPICS API one-shot and multi-shot
    for multi-shot: pass output (topics + ground trtuh) of previous epoch,
    returned array is not shuffled randomly to make it easier to keep track for
    the consecutive calls for multi-shot. This has no further impact as
    denoising is done as if this array was shuffled.
    """
    if np.random.random() < p:
        topic = np.random.choice(taxonomy_ids, 1)[0]
        ground_truth = 0  # noisy
    else:
        topic = np.random.choice(top_topics, 1)[0]
        ground_truth = 1  # genuine
    return topic, ground_truth


def denoise(topics_view, unobserved_topics):
    nb_epochs = len(topics_view)
    genuine_topics = []
    noisy_topics = []
    observed_topics_not_noisy = []

    for epoch in range(nb_epochs):
        temp_view = topics_view[0 : epoch + 1]
        # check if duplicates as they would be genuine
        temp_genuine = list(set([x for x in temp_view if temp_view.count(x) > 1]))
        genuine_topics.append(list(set(temp_genuine)))
        temp_noisy = []
        # check for remaining topics if never observed as they would be noisy
        for topic in list(set(temp_view).difference(set(temp_genuine))):
            if topic in unobserved_topics:
                temp_noisy.append(topic)
        noisy_topics.append(list(set(temp_noisy)))
        # difference of view and noisy = observed not noisy
        temp_observed_not_noisy = list(set(temp_view).difference(set(temp_noisy)))
        observed_topics_not_noisy.append(temp_observed_not_noisy)
    return genuine_topics, noisy_topics, observed_topics_not_noisy


def compare_truth_denoise(topics_view, ground_truth, noisy_topics):
    """
    Compute true positive, false positive, etc. positive class = noisy / negative
    class = genuine
    """
    denoise_pred = [1] * len(topics_view)  # init to 1 = not noisy
    # go through, if in noisy list, set to 0

    for i in range(len(topics_view)):
        if topics_view[i] in noisy_topics:
            denoise_pred[i] = 0

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    assert len(ground_truth) == len(denoise_pred)
    for i in range(len(ground_truth)):
        if ground_truth[i] == denoise_pred[i]:
            # positive class = noisy | negative class = genuine
            if ground_truth[i] == 0:
                tp += 1
            else:
                tn += 1
        else:
            if ground_truth[i] == 0:
                fp += 1
            else:
                fn += 1
    # print(topics_view, ground_truth, noisy_topics, denoise_pred, tp, fp, tn, fn)
    return tp, fp, tn, fn


def aggregate_denoise_results(results, print_results, output_file):
    """Aggregate the results returned by the pool of workers"""
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    intersection = []

    for result in results:
        true_positive += result[0]
        false_positive += result[1]
        true_negative += result[2]
        false_negative += result[3]
        intersection.append(result[4])

    accuracy = (true_positive + true_negative) / (
        true_positive + true_negative + false_positive + false_negative
    )
    precision = (true_positive) / (true_positive + false_positive)
    true_positive_rate = (true_positive) / (true_positive + false_negative)
    false_positive_rate = (false_positive) / (false_positive + true_negative)

    df = pd.DataFrame({"intersection": np.array(intersection)})
    if print_results:
        with open(output_file, "a") as f:
            f.write("true_positive: {}\n".format(true_positive))
            f.write("false_positive: {}\n".format(false_positive))
            f.write("true_negative: {}\n".format(true_negative))
            f.write("false_negative: {}\n".format(false_negative))
            f.write("Accuracy: {}\n".format(accuracy))
            f.write("precision: {}\n".format(precision))
            f.write("recall/TPR: {}\n".format(true_positive_rate))
            f.write("FPR: {}\n".format(false_positive_rate))
            f.write("--")
            f.write("{}\n".format(df.describe()))

    return (
        accuracy,
        precision,
        true_positive_rate,
        false_positive_rate,
        df["intersection"].min(),
        df["intersection"].median(),
        df["intersection"].max(),
    )


def aggregate_re_identification_results(results, epoch_index, output_prefix):
    size_reidentified_groups = np.array(results)
    np.save(
        output_prefix + "_epoch_" + str(epoch_index) + "_size_reidentified_groups.npy",
        size_reidentified_groups,
    )
    nb_users = len(results)

    nb_users_re_identified = sum(size_reidentified_groups == 1)
    nb_users_failure = sum(size_reidentified_groups == 0)
    nb_users_better_chance = sum(size_reidentified_groups > 1)
    if epoch_index == 0:
        mode = "w"
    else:
        mode = "a"
    with open(output_prefix + "_cdf_reidentification.stats", mode) as f:
        f.write("Epoch --- {} ---\n".format(epoch_index))
        f.write(
            "Uniquely re-identified: {}  -  {}\n".format(
                nb_users_re_identified, nb_users_re_identified / nb_users
            )
        )
        f.write(
            "Failure re-identified: {}  -  {}\n".format(
                nb_users_failure, nb_users_failure / nb_users
            )
        )
        f.write(
            "Better chance re-identified: {}  -  {}\n".format(
                nb_users_better_chance, nb_users_better_chance / nb_users
            )
        )


#### Experiments


## Init Experiment
def init_exp_init_worker(unobserved_topics, taxonomy_ids, nb_epochs):
    global shared_unobserved_topics
    shared_unobserved_topics = unobserved_topics
    global shared_taxonomy_ids
    shared_taxonomy_ids = taxonomy_ids
    global shared_nb_epochs
    shared_nb_epochs = nb_epochs


def init_exp_pool_function_a(user, p=0.05):
    global shared_unobserved_topics
    global shared_taxonomy_ids
    global shared_nb_epochs
    return user.init_experiments_a(
        shared_nb_epochs, shared_taxonomy_ids, shared_unobserved_topics, p
    )


def init_exp_pool_function_b(user, p=0.05):
    global shared_unobserved_topics
    global shared_taxonomy_ids
    global shared_nb_epochs
    return user.init_experiments_b(
        shared_nb_epochs, shared_taxonomy_ids, shared_unobserved_topics, p
    )


def init_exp_a(users, unobserved_topics, taxonomy_ids, nb_epochs, p=0.05):
    with Pool(
        initializer=init_exp_init_worker,
        initargs=(unobserved_topics, taxonomy_ids, nb_epochs),
    ) as pool:
        results = pool.starmap(init_exp_pool_function_a, zip(users, repeat(p)))
    # collect results and update
    for r in results:
        id = r[0]
        users[id].a_topics_view = r[1]
        users[id].a_ground_truth = r[2]
        users[id].a_genuine_topics = r[3]
        users[id].a_noisy_topics = r[4]
        users[id].a_observed_topics = r[5]


def init_exp_b(users, unobserved_topics, taxonomy_ids, nb_epochs, p=0.05):
    with Pool(
        initializer=init_exp_init_worker,
        initargs=(unobserved_topics, taxonomy_ids, nb_epochs),
    ) as pool:
        results = pool.starmap(init_exp_pool_function_b, zip(users, repeat(p)))
    # collect results and update
    for r in results:
        id = r[0]
        users[id].b_topics_view = r[1]
        users[id].b_ground_truth = r[2]
        users[id].b_genuine_topics = r[3]
        users[id].b_noisy_topics = r[4]
        users[id].b_observed_topics = r[5]


## Denoise Experiment
def denoise_exp_specific_epoch_pool_function_a(user, epoch_index):
    tp, fp, tn, fn = compare_truth_denoise(
        user.a_topics_view[0 : epoch_index + 1],
        user.a_ground_truth[0 : epoch_index + 1],
        user.a_noisy_topics[epoch_index],
    )

    user_interests = set()
    for epoch in range(epoch_index):
        top_epoch = user.epoch_topics[epoch]
        for t in top_epoch:
            user_interests.add(t)

    intersection = list(
        set(user_interests).intersection(user.a_genuine_topics[epoch_index])
    )

    return tp, fp, tn, fn, len(intersection)


def denoise_exp_specific_epoch_pool_function_b(user, epoch_index):
    tp, fp, tn, fn = compare_truth_denoise(
        user.b_topics_view[0 : epoch_index + 1],
        user.b_ground_truth[0 : epoch_index + 1],
        user.b_noisy_topics[epoch_index],
    )

    user_interests = set()
    for epoch in range(epoch_index):
        top_epoch = user.epoch_topics[epoch]
        for t in top_epoch:
            user_interests.add(t)

    intersection = list(
        set(user_interests).intersection(user.b_genuine_topics[epoch_index])
    )

    return tp, fp, tn, fn, len(intersection)


def denoise_exp_specific_epoch_a(users, epoch_index, output_prefix):
    with Pool() as pool:
        results = pool.starmap(
            denoise_exp_specific_epoch_pool_function_a,
            zip(users, repeat(epoch_index)),
        )
    # collect results
    output_file = output_prefix + "_denoise_a.stats"
    if epoch_index == 0:  # we overwrite in case file remaining from previous run
        with open(output_file, "w") as f:
            f.write("--Epoch: {} ---\n".format(epoch_index))
    else:
        with open(output_file, "a") as f:
            f.write("--Epoch: {} ---\n".format(epoch_index))
    return aggregate_denoise_results(results, True, output_file)


def denoise_exp_specific_epoch_b(users, epoch_index, output_prefix):
    with Pool() as pool:
        results = pool.starmap(
            denoise_exp_specific_epoch_pool_function_b,
            zip(users, repeat(epoch_index)),
        )
    # collect results
    output_file = output_prefix + "_denoise_b.stats"
    if epoch_index == 0:  # we overwrite in case file remaining from previous run
        with open(output_file, "w") as f:
            f.write("--Epoch: {} ---\n".format(epoch_index))
    else:
        with open(output_file, "a") as f:
            f.write("--Epoch: {} ---\n".format(epoch_index))
    return aggregate_denoise_results(results, True, output_file)


def denoise_exp_all_epochs_a(users, nb_epochs, output_prefix):
    accuracy = np.zeros(nb_epochs)
    precision = np.zeros(nb_epochs)
    tp_rate = np.zeros(nb_epochs)
    fp_rate = np.zeros(nb_epochs)
    inter_mins = np.zeros(nb_epochs)
    inter_meds = np.zeros(nb_epochs)
    inter_maxs = np.zeros(nb_epochs)
    for i in range(nb_epochs):
        (
            accuracy[i],
            precision[i],
            tp_rate[i],
            fp_rate[i],
            inter_mins[i],
            inter_meds[i],
            inter_maxs[i],
        ) = denoise_exp_specific_epoch_a(users, i, output_prefix)
    np.save(output_prefix + "_denoise_a_accuracy.npy", accuracy)
    np.save(output_prefix + "_denoise_a_precision.npy", precision)
    np.save(output_prefix + "_denoise_a_tpr.npy", tp_rate)
    np.save(output_prefix + "_denoise_a_fpr.npy", fp_rate)
    np.save(output_prefix + "_denoise_a_intersection_min.npy", inter_mins)
    np.save(output_prefix + "_denoise_a_intersection_med.npy", inter_meds)
    np.save(output_prefix + "_denoise_a_intersection_max.npy", inter_maxs)


def denoise_exp_all_epochs_b(users, nb_epochs, output_prefix):
    accuracy = np.zeros(nb_epochs)
    precision = np.zeros(nb_epochs)
    tp_rate = np.zeros(nb_epochs)
    fp_rate = np.zeros(nb_epochs)
    inter_mins = np.zeros(nb_epochs)
    inter_meds = np.zeros(nb_epochs)
    inter_maxs = np.zeros(nb_epochs)
    for i in range(nb_epochs):
        (
            accuracy[i],
            precision[i],
            tp_rate[i],
            fp_rate[i],
            inter_mins[i],
            inter_meds[i],
            inter_maxs[i],
        ) = denoise_exp_specific_epoch_b(users, i, output_prefix)
    np.save(output_prefix + "_denoise_b_accuracy.npy", accuracy)
    np.save(output_prefix + "_denoise_b_precision.npy", precision)
    np.save(output_prefix + "_denoise_b_tpr.npy", tp_rate)
    np.save(output_prefix + "_denoise_b_fpr.npy", fp_rate)
    np.save(output_prefix + "_denoise_b_intersection_min.npy", inter_mins)
    np.save(output_prefix + "_denoise_b_intersection_med.npy", inter_meds)
    np.save(output_prefix + "_denoise_b_intersection_max.npy", inter_maxs)


## Re-identification Experiment


def reidentification_exp_pool_function(
    view_a_user, user_id_ground_truth, view_b_users, reidentification_dict
):
    candidates = []
    for t_a in view_a_user:
        for user_id_b in reidentification_dict[t_a]:
            candidates.append(user_id_b)

    candidates_ids = list(set(candidates))
    hamming_distances = []
    min_distance = 1000  # init to big number for now
    for c_id in candidates_ids:
        union = set(view_a_user).union(set(view_b_users[c_id]))
        intersection = set(view_a_user).intersection(set(view_b_users[c_id]))
        difference = union.difference(intersection)
        h_d = len(difference)
        hamming_distances.append(h_d)
        if h_d < min_distance:
            min_distance = h_d

    reidentified_ids = []
    for i in range(len(hamming_distances)):
        if hamming_distances[i] == min_distance:
            reidentified_ids.append(candidates_ids[i])

    if user_id_ground_truth not in reidentified_ids:
        return 0
    else:
        return len(reidentified_ids)


def reidentification_exp_specific_epoch(
    users, epoch_index, taxonomy_ids, output_prefix
):
    reidentification_dict = {}
    for t_id in taxonomy_ids:
        reidentification_dict[t_id] = []

    view_a = []
    view_b = []
    user_ids_ground_truth = []
    for user in users:
        view_a.append(user.a_observed_topics[epoch_index])
        user_ids_ground_truth.append(user.id)
        view_b.append(user.b_observed_topics[epoch_index])
        for topic in view_b[-1]:
            reidentification_dict[topic].append(user.id)
    with Pool() as pool:
        results = pool.starmap(
            reidentification_exp_pool_function,
            zip(
                view_a,
                user_ids_ground_truth,
                repeat(view_b),
                repeat(reidentification_dict),
            ),
        )
    aggregate_re_identification_results(results, epoch_index, output_prefix)

    return


def reidentification_exp_all_epochs(users, nb_epochs, taxonomy_ids, output_prefix):
    for epoch in range(nb_epochs):
        reidentification_exp_specific_epoch(users, epoch, taxonomy_ids, output_prefix)
