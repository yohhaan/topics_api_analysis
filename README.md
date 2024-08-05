# topics_api_analysis

This is the code artifact of the paper [A Public and Reproducible Assessment of the Topics API on Real Data](https://arxiv.org/abs/2403.19577)

```bibtex
@inproceedings{topics_secweb24_beugin,
      title={A Public and Reproducible Assessment of the Topics API on Real Data},
      author={Yohan Beugin and Patrick McDaniel},
      booktitle={2024 IEEE Security and Privacy Workshops (SPW)},
      year={2024},
      month={may},
}
```

Check out also our other
[topics_analysis](https://github.com/yohhaan/topics_analysis) repository.

---
## Getting Started

1. Clone this [topics_api_analysis](https://github.com/yohhaan/topics_api_analysis)
   repository and the
   [topics_classifier](https://github.com/yohhaan/topics_classifier)
   submodule at once with:
   - `git clone --recurse-submodules git@github.com:yohhaan/topics_api_analysis.git` (SSH)
   - `git clone --recurse-submodules
     https://github.com/yohhaan/topics_api_analysis.git` (HTTPS)

A `Dockerfile` is provided under `.devcontainer/`; for direct integration with
VS Code or to manually build the image and deploy the Docker container, follow
the instructions in this [guide](https://gist.github.com/yohhaan/b492e165b77a84d9f8299038d21ae2c9).

## Reproduction Steps

**Topics classification:** refer to and execute the bash scripts in the
corresponding folder under [`./data`](./data) to classify the different
datasets with the Topics API:

- CrUX: `cd data/crux && ./crux.sh`
- Tranco: `cd data/tranco && ./tranco.sh`
- Real Browsing Histories: `cd data/web_data && ./web_data.sh`

**Topics evaluation:** refer to the
[`topics_simulator.py`](topics_simulator.py) script to evaluate the Topics API
(simulation of the API for users, denoising, and re-identification across epochs)
```
usage: python3 topics_simulator.py [-h]
                                   users_topics_tsv nb_epochs config_model_json top_list_tsv
                                   unobserved_topics_threshold repeat_each_user_n_times output_prefix

Simulate the Topics API and evaluate its privacy guarantees

positional arguments:
  users_topics_tsv
  nb_epochs
  config_model_json
  top_list_tsv
  unobserved_topics_threshold
  repeat_each_user_n_times
  output_prefix
```

Examples:
- `python3 topics_simulator.py data/web_data/users_topics_5_weeks.tsv 5 topics_classifier/chrome5/config.json data/crux/crux_202406_chrome5_topics-api.tsv 10 1 data/reidentification_exp/5_weeks_10_unobserved`
- `python3 topics_simulator.py data/web_data/users_topics_5_weeks.tsv 5 topics_classifier/chrome5/config.json data/crux/crux_202406_chrome5_topics-api.tsv 10 100 data/denoise_exp/5_weeks_100_repetitions_10_unobserved`

**Analysis:** to extract statistics and plot the figures, refer to the
[`analysis.py`](analysis.py) script.
