# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 21:29:55 2021

@author: Steve
"""

import xml.etree.ElementTree as ET
import os
import glob

img_folder = './OIDv4_ToolKit-master/OID/Dataset/train/phone'


   

for xml_file in glob.glob(img_folder+'/*.xml'):

    try:
        tree=ET.parse(open(xml_file))
        root = tree.getroot()
        
        for stuff in root.findall('object'):
            
            stuff = stuff.find('name')
            if stuff.text == 'Vehicle_registration_plate':
                print(xml_file)
    except:
        print(f'BAD FILE IS: {xml_file}')
            
            
            
            
        #stuff.text = 'Windscreen'
        #print(stuff)
    tree.write(xml_file)

os.getcwd()


#Test on single file
tree=ET.parse(open('./OIDv4_ToolKit-master/OID/Dataset/train/windscreen\h82.xml'))
root = tree.getroot()
for stuff in root.findall('object'):
    
    stuff = stuff.find('name')
    stuff.text = 'Windscreen'
    print(stuff)
tree.write('./OIDv4_ToolKit-master/OID/Dataset/train/windscreen\h82.xml')
    
    
