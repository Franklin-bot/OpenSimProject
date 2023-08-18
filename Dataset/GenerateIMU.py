import pandas as pd
import numpy as np
import quaternion
from bs4 import BeautifulSoup
from nested_lookup import nested_lookup, get_all_keys
from itertools import islice
from IPython.display import display
import os

def rotateIMU(q, std_dev):
    theta = np.random.normal(loc=0, scale=std_dev)
    axis = np.random.random(size=3)
    axis = axis/np.linalg.norm(axis)
    q_rotation = np.quaternion(np.cos(theta/2), *(np.sin(theta/2) * axis))

    return q * q_rotation

def translateIMU(file_path):
    f = open(file_path, "r")
    data = f.read()

    

translateIMU("/Users/FranklinZhao/OpenSimProject/Dataset/Rajagopal2015_Marked_No_Muscles.xml")






