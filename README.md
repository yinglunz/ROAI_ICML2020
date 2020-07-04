This is the python code for Robust Outlier Arm Identification accepted at ICML 2020. Used packages include: numpy, math, multiprocessing, copy, functools, astropy.stats, time, datetime and matplotlib.

ROAI_class.py: These contain classes that are imported by other .py files.

termination_count.py: This contains code to plot Fig 1, in Section 7.1. It computes the empirical stopping time and compare it with the theoretical sample complexity upper bound.

OAI_comparison.py: This contains code for Fig. 2 and Fig. 3 in Section 7.2. It computes |proposed threshold - true threshold| or sample complexity upper bounds for different contamination levels.

ROAI.py: This contains code for Fig. 4 in Section 7.3. It compares the anytime performance of different algorithms on simulated data.

real_data.py: This contains code for Fig. 5 in Section 7.3. It compares the anytime performance of different algorithms on the wine dataset. 

wine.mat: This file contains the raw wine quality dataset; the preprocessed dataset is included in real_data.py.

