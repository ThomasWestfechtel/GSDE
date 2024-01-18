# GSDE
PyTorch Code for GSDE - Gradual Source Domain Expansion for Unsupervised Domain Adaptation (WACV 2024)
https://openaccess.thecvf.com/content/WACV2024/papers/Westfechtel_Gradual_Source_Domain_Expansion_for_Unsupervised_Domain_Adaptation_WACV_2024_paper.pdf

### Method overview
Unsupervised domain adaptation (UDA) tries to overcome the need for a large labeled dataset by transferring knowledge from a source dataset, with lots of labeled data, to a target dataset, that has no labeled data. Since there are no labels in the target domain, early misalignment might propagate into the later stages and lead to an error build-up.
In order to overcome this problem, we propose a gradual source domain expansion (GSDE) algorithm. GSDE trains the UDA task several times from scratch, each time reinitializing the network weights, but each time expands the source dataset with target data. In particular, the highest-scoring target data of the previous run are employed as pseudo-source samples with their respective pseudo-label. Using this strategy, the pseudo-source samples induce knowledge extracted from the previous run directly from the start of the new training. This helps align the two domains better, especially in the early training epochs.
In this study, we first introduce a strong baseline network and apply our GSDE strategy to it. We conduct experiments and ablation studies on three benchmarks (Office-31, OfficeHome, and DomainNet) and outperform state-of-the-art methods. We further show that the proposed GSDE strategy can improve the accuracy of a variety of different state-of-the-art UDA approaches.

### Usage
To train the network on Office31:
bash train_o31.sh

To train the network on OfficeHome:
bash train_oh.sh

To train the network on DomainNet:
bash train_dn.sh

To change the adaptation task -> Change index in the script file

Change the 2 lines in the script to where the datasets are stored:

    s_dset_path='../Office31/'${source[index]}'/label_c.txt'
    
    t_dset_path='../Office31/'${target[index]}'/label_c.txt'

Example for how the label_c.txt file looks like can be seen for Office31 in the datasets folder

### Citation
If you use GSDE code please cite:
```text
@inproceedings{westfechtel2024gradual,
  title={Gradual Source Domain Expansion for Unsupervised Domain Adaptation},
  author={Westfechtel, Thomas and Yeh, Hao-Wei and Zhang, Dexuan and Harada, Tatsuya},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1946--1955},
  year={2024}
}
```
