#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose: Convert shell script for CaMa-Flood model simulation into structured, strongly-typed Python
Licensed under the Apache License, Version 2.0.
"""
import os
import shutil
import glob


os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


def run_config_1(config: dict):
    """
       Run CaMa-Flood simulations using config parameters provided in a dictionary.

       Args:
           config (dict): Configuration dictionary containing all simulation parameters.
   """

    # ================================================
    # (2) Initial setting
    config["NMLIST"]      =         config["RDIR"]   +   config["EXP"]    +   config["NMLIST"]
    config["CSETFILE"]      =       config["NMLIST"]
    config["LOGOUT"]        =       config["RDIR"]   +   config["EXP"]    +   config["LOGOUT"]
    config['CDIMINFO']      =       config['FMAP']   +   config["CDIMINFO"]
    config['CNEXTXY']       =       config['FMAP']   +   config['CNEXTXY']
    config['CGRAREA']       =       config['FMAP']   +   config['CGRAREA']
    config['CELEVTN']       =       config['FMAP']   +   config['CELEVTN']
    config['CNXTDST']       =       config['FMAP']   +   config['CNXTDST']
    config['CRIVLEN']       =       config['FMAP']   +   config['CRIVLEN']
    config['CFLDHGT']       =       config['FMAP']   +   config['CFLDHGT']
    config['CRIVWTH']       =       config['FMAP']   +   config['CRIVWTH']
    config['CRIVHGT']       =       config['FMAP']   +   config['CRIVHGT']
    config['CRIVMAN']       =       config['FMAP']   +   config['CRIVMAN']
    config['CPTHOUT']       =       config['FMAP']   +   config['CPTHOUT']
    config['CINPMAT']       =       config['FMAP']   +   config['CINPMAT']
    config['COUTDIR']       =       config['COUTDIR']+   config['EXP']


    # (2a) Create and enter the run directory
    os.makedirs         (config["RDIR"]     +   config["EXP"], exist_ok=True)
    # os.chdir            (config["RDIR"]     +   config["EXP"])

    # (2b) for new simulation, remove old files in running directory
    if config["SPINUP"] == 0:
        for pattern in ["????-sp*", "*.bin", "*.pth", "*.vec", "*.nc", "*.log", "*.txt", "restart*"]:
            for file in glob.glob(pattern):
                try:
                    shutil.rmtree(config["RDIR"]   +   file, ignore_errors=True)
                except IsADirectoryError:
                    os.remove(config["RDIR"]   +   file)
    else:
        config  ["NSP"]     =       0         # Disable spinup count if restarting

    # ================================================
    # (3) For each simulation year, modify setting
    # #--  loop 1-year simulation from $YSTART to $YEND
    config  ["ISP"]                     =       1
    config  ["IYR"]                     =       config["YSTA"]


    return config