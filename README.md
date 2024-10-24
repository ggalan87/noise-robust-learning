# Noise-robust learning
The scripts which correspond to the experiments reported in the paper can be found in [experiments scripts folder](lightning/cli_pipelines/experiments_scripts). All experiments are mostly built around [lighning](./lightning) package which contains models, datasets etc. Data denoising methods can be found in [torch_metric_learning](./torch_metric_learning) package which is essentially an extension to [PML](https://github.com/KevinMusgrave/pytorch-metric-learning).

# Dependencies
The repository is mostly built around [pytorch lightning](https://github.com/Lightning-AI/pytorch-lightning) framework and [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) library.

Parts of the repository includes code from other sources. Every such part has link to its source, e.g. url to the relevant implementation.

The [requirements](./requirements.txt) file includes the dependencies of the project, although some are deprecated, e.g. initial repo was built around mmcv and relevant packages which are not needed by the paper's code.


## TODO
* Refactor files and packages
* Upload experiments scripts
* Provide / improve documentation
* Remove custom paths from test/example files
* Migrate to newer dependencies versions
* Migrate to poetry for dependency management

## Relevant publications
This repository contains the code of the following papers.

```latex
@inproceedings{galanakis2024noise,
  title={Noise-robust person re-identification through nearest-neighbor sample filtering},
  author={Galanakis, George and Zabulis, Xenophon and Argyros, Antonis A},
  booktitle={2024 IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
  pages={1--8},
  year={2024},
  organization={IEEE}
}

@inproceedings{Galanakis2024,
	author = {Galanakis, George and Zabulis, Xenophon and Argyros, Antonis A},
	title = {Nearest neighbor-based data denoising for deep metric learning},
	booktitle = {VISAPP},
	year = {2024},
}
```