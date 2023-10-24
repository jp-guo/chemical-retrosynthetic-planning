# Retrosynthetic-Planning

Dataset can be downloaded at [link](https://jbox.sjtu.edu.cn/l/y1BtE5).

## Get the environment

```bash
conda create -n ml-proj python=3.9
conda activate ml-proj
pip install -r requirements.txt
```

## File structure

```
D:.
│  .gitignore
│  LICENSE
│  README.md
│  utils.py
│
└─data
    ├─MoleculeEvaluationData
    │      test.pkl.gz
    │      train.pkl.gz
    │
    ├─Multi-Step task
    │      starting_mols.pkl.gz
    │      target_mol_route.pkl
    │      test_mols.pkl
    │
    ├─rdchiral
    │  │  .gitignore
    │  │  LICENSE
    │  │  README.md
    │  │  setup.py
    │  │
    │  ├─build
    │  │  └─bdist.win-amd64
    │  ├─dist
    │  │      rdchiral-0.0.0-py3.9.egg
    │  │
    │  ├─rdchiral(Not used)
    │  │  │  bonds.py
    │  │  │  chiral.py
    │  │  │  clean.py
    │  │  │  initialization.py
    │  │  │  main.py
    │  │  │  template_extractor.py
    │  │  │  utils.py
    │  │  │  __init__.py
    │  │  │
    │  │  ├─backup
    │  │  │      bonds.py
    │  │  │      chiral.py
    │  │  │      clean.py
    │  │  │      initialization.py
    │  │  │      main.py
    │  │  │      template_extractor.py
    │  │  │      utils.py
    │  │  │      __init__.py
    │  │  │
    │  │  ├─old
    │  │  │      chiral.py
    │  │  │      clean.py
    │  │  │      initialization.py
    │  │  │      main.py
    │  │  │      template_extractor.py
    │  │  │      utils.py
    │  │  │      __init__.py
    │  │  │
    │  │  └─test
    │  │          test_smiles_from_50k_uspto.txt
    │  │          __init__.py
    │  │
    │  ├─rdchiral.egg-info
    │  │      dependency_links.txt
    │  │      PKG-INFO
    │  │      SOURCES.txt
    │  │      top_level.txt
    │  │
    │  ├─templates
    │  │      clean_and_extract_uspto.py
    │  │      Examine templates.ipynb
    │  │      example_template_extractions_bad.json
    │  │      example_template_extractions_good.json
    │  │      README.md
    │  │
    │  └─test
    │          Test rdchiral notebook.ipynb
    │          test_rdchiral.py
    │          test_rdchiral_cases.json
    │
    └─schneider50k
            raw_test.csv
            raw_train.csv
            raw_val.csv

```

## Task1:Single-step retrosynthesis prediction

In this part, we investigate the performance of MLP, SVM and [GLN](https://github.com/potus28/ML-Project---GLN) for
single-step
retrosynthesis prediction.

## Setup

To prepare the environments for GLN, please refer to [GLN Setup](single_step/gln/README.md) for detaild information.

Run the following command to build the Morgan Fingerprint dataset:

```angular2html
cd single_step/dropbox
python dataset.py --build
cd -
```

Run the following command to train the GLN:

```angular2html
cd single_step/exps
bash scripts/train_gln.sh
```

Reproduce the results of GLN in our project:

```angular2html
bash scripts/eval_gln.sh
```

Run the following command to train an MLP:

```angular2html
bash scripts/train_mlp.sh
```

Reproduce the results of MLP in our project:

```angular2html
bash scripts/eval_mlp.sh
```

Evaluate the performance of SVM with the following command:

```angular2html
bash scripts/train_svm.sh
```

Reproduce the results:

```angular2html
bash scripts/eval_svm.sh
```

## Task2: Molecule evaluation

This is a prediction task to predict the synthetic cost of the given molecule. Training and test data are provided in
the format of (Packed Morgan FingerPrint, cost). Design your model to predict the cost for each molecule. (unpack the
FingerPrint with numpy.unpackbits). What’s more, sometimes we need to predict the synthesis cost of multiple molecules
simultaneously. One way to implement this is to predict each molecule separately and then sum them up:

$$
cost = f(m_1) + f(m_2) + ... + f(m_k)
$$

Another way is to design a function to predict the cost of multiple molecules:

$$
cost = f(m_1 + m_2 + ... + m_k)
$$

The synthetic cost of one molecule is mainly determined by its composed atoms and structure. Morgan FingerPrint can
describe the atoms and structures of given molecule. So, we assume that molecules with similar fingerprints have similar
synthetic cost. Discuss which method is better to estimate cost of multiple molecules, Equation 3.1 or Equation 3.2.

For the situation of Equation 3.2, graph neural network might be helpful. Cosine similarity of the fingerprints can be
used as the metric to describe the relationship between molecules. Alternatively, you can try to establish connections
between molecules in other ways.

Run the following command to build the molecule evaluation dataset:

```angular2html
cd molecule_eval
python main.py
```

## Task3: Multi-step retrosynthesis planning

In general, molecules cannot be synthesized in one step and multiple reactions are required.

Therefore, an efficient search algorithm is needed to find such a synthetic route with the help of single step
retrosynthesis model and molecule evaluation function.

In this task, a set of target molecules to be synthesized and a set of starting molecules are provided.

You need to design a search algorithm to find the reactions, which can synthetic target molecules with the starting
molecules.

<!-- Figure 1 provide an example of successful synthetic route. -->