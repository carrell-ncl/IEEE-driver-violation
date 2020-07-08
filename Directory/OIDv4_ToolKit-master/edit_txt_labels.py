#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:17:22 2020

@author: carrell
"""

import os

path = './OIDv4_ToolKit-master/OID/Dataset/train/'
b = os.getcwd()

DIR = os.path.join(path, "Mobile_phone", "Label")


for file in os.listdir(DIR):
    if file.endswith('.txt'):     
        with open(os.path.join(DIR, file), 'r') as tfile:
            filedata = tfile.read()
            #Select what text you want to edit
            filedata = filedata.replace('Mobile p', 'P')
        with open(os.path.join(DIR, file), 'w') as tfile:
            tfile.write(filedata)
            
        


    

