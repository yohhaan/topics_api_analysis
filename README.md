# topics_api_analysis

This is the code artifact of the paper [A Public and Reproducible Assessment of the Topics API on Real Data](https://arxiv.org/abs/2403.19577)

```bibtex
@inproceedings{topics_secweb24_beugin,
      title={A Public and Reproducible Assessment of the Topics API on Real Data},
      author={Yohan Beugin and Patrick McDaniel},
      booktitle={2024 IEEE Security and Privacy Workshops (SPW)},
      year={2024},
      month={may},
      doi={10.48550/arXiv.2403.19577}
}
```

Check out also our other [topics_analysis](https://github.com/yohhaan/topics_analysis) repository and [PETS'24 paper](https://doi.org/10.56553/popets-2024-0004):

```bibtex
@inproceedings{topicsweb24_beugin,
      title={Interest-disclosing Mechanisms for Advertising are Privacy-Exposing (not Preserving)},
      author={Yohan Beugin and Patrick McDaniel},
      booktitle={Proceedings on {Privacy} {Enhancing} {Technologies} {Symposium} ({PETS})},
      year={2024},
      month={july},
      doi={10.56553/popets-2024-0004}
}
```

---
## Getting Started

- Clone this [topics_api_analysis](https://github.com/yohhaan/topics_api_analysis) repository and the [topics_classifier](https://github.com/yohhaan/topics_classifier)
   submodule at once with:
   - `git clone --recurse-submodules git@github.com:yohhaan/topics_api_analysis.git` (SSH)
   - `git clone --recurse-submodules https://github.com/yohhaan/topics_api_analysis.git` (HTTPS)

- Note: the [`.devcontainer/`](.devcontainer/) directory contains the config for integration with VS Code (see [guide here](https://github.com/PoPETS-AEC/examples-and-other-resources/blob/main/resources/vs-code-docker-integration.md)).

- Then, follow either set of instructions (or install dependencies manually):

> <details><summary>Using the Docker image from the Container Registry</summary>
>
> This [GitHub workflow](.github/workflows/build-push-docker-image.yaml)
> automatically builds and pushes the Docker image to GitHub's Container Registry
> when the `Dockerfile` or the `requirements.txt` files are modified.
>
> 1. Pull the Docker image:
> ```bash
> docker pull ghcr.io/yohhaan/topics_api_analysis:main
> ```
> 2. Launch the Docker container, attach the current working directory (i.e.,
> run from the root of the cloned git repository) as a volume, set the context
> to be that volume, and provide an interactive bash terminal:
> ```bash
> docker run --rm -it -v ${PWD}:/workspaces/topics_api_analysis \
>     -w /workspaces/topics_api_analysis \
>     --entrypoint bash ghcr.io/yohhaan/topics_api_analysis:main
> ```
> </details>


> <details><summary>Using a locally built Docker image</summary>
>
> 1. Build the Docker image:
> ```bash
> docker build -t topics_api_analysis:main .
> ```
> 2. Launch the Docker container, attach the current working directory (i.e.,
> run from the root of the cloned git repository) as a volume, set the context
> to be that volume, and provide an interactive bash terminal:
> ```bash
> docker run --rm -it -v ${PWD}:/workspaces/topics_api_analysis \
>     -w /workspaces/topics_api_analysis \
>     --entrypoint bash topics_api_analysis:main
> ```
> </details>

## Reproduction Steps

**Topics classification:** refer to and execute the bash scripts in the corresponding folder under [`./data`](./data) to classify the different datasets with the Topics API:

- CrUX: `cd data/crux && ./crux.sh`
- Tranco: `cd data/tranco && ./tranco.sh`
- Real Browsing Histories: `cd data/web_data && ./web_data.sh`

**Topics evaluation:** refer to the [`topics_simulator.py`](topics_simulator.py) script to evaluate the Topics API (simulation of the API for users, denoising, and re-identification across epochs)
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

**Analysis:** to extract statistics and plot the figures, refer to the [`analysis.py`](analysis.py) script.
