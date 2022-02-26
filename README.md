# Robust Outlier Arm Identification

This repository contains the python code for our ICML 2020 paper **Robust Outlier Arm Identification**. Used packages include: numpy, math, multiprocessing, copy, functools, astropy.stats, time, datetime and matplotlib.

Use the following command to reproduce our experiment in Figure 1.(b).

```
python3 termination_count.py
```

Use the following command to reproduce our experiments in Figure 2 and Figure 3 (set `contamination_level = 0` in `OAI_comparison.py`).

```
python3 OAI_comparison.py
```

Use the following command to reproduce our experiment in Figure 4.

```
python3 ROAI.py
```

Take the following steps to reproduce our experiment in Figure 5. First, get dataset `wine.mat` from this [website](http://odds.cs.stonybrook.edu/wine-dataset/). Next, preprocess the dataset based on the experiment description. Then, input the preprocessed means of normal and outlier arms into `y_normal` and `y_outlier` (in file `real_data.py`). Finally, run the following command.

```
python3 real_data.py
```
