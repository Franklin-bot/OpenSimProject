from vmdpy import VMD
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import islice
import pandas as pd
from tqdm import tqdm
import scipy.interpolate as interpolate
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import minmax_scale
from scipy import signal