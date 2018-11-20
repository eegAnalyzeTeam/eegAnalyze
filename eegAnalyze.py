# This program is where the main function at. 
# The program calls preprocessing functions, calculate psds for each
# person and use anova test to find the significant channels and sub-frequencies
import mne
import numpy as np
from matplotlib import pyplot as plt
import os, sys

