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
from datetime import datetime
import time
import re

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


def run_config_3(start_time, config: dict):
    """
       Run CaMa-Flood simulations using config parameters provided in a dictionary.

       Args:
           config (dict): Configuration dictionary containing all simulation parameters.
   """
    print(f"Elapsed time: str({time.time() - start_time:.2f}) seconds")
    print("end:" +  str(config["SYEAR"]) +  str(datetime.now()))

    if  os.path.exists(config["LOGOUT"]):
        shutil.move(config["LOGOUT"], f"{config['RDIR']}{config['EXP']}log_CaMa-{config['CYR']}.txt")
    # ================================================
    # (6) manage spin up

    # if curent spinup time $ISP < required spinup time $NSP
    #   copy the restart file restart$(IYR+1) to restart${IYR}
    #   copy the outputs to directory "${IYR}-sp1"

    config["SPINUP"]             =           1

    if config["IYR"]          ==    config["YSTA"]:
        if config["ISP"]      <=    config["NSP"]:
            config["IYR1"]    =     config["IYR"] + 1
            IYR1              =     config["IYR1"]

            config["CYR1"]    =     f"{IYR1:04d}"
            CYR1              =     config["CYR1"]
            CYR               =     config["CYR"]
            ISP               =     config["ISP"]

            src               =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR1}010100.bin")
            dst_sp            =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR}010100.bin-sp{ISP}")
            dst               =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR}010100.bin")

            if os.path.exists(src):
                shutil.copy2(src, dst_sp)
                shutil.move(src, dst)

            src               =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR1}010100.bin.pth")
            dst_sp            =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR}010100.bin.pth-sp{ISP}")
            dst               =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR}010100.bin.pth")
            if os.path.exists(src):
                shutil.copy2(src, dst_sp)
                shutil.move(src, dst)

            src               =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR1}010100.nc")
            dst_sp            =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR}010100.nc-sp{ISP}")
            dst               =     os.path.join(config['RDIR'] + config['EXP']  + config["CVNREST"] +  f"{CYR}010100.nc")
            if os.path.exists(src):
                shutil.copy2(src, dst_sp)
                shutil.move(src, dst)

            outdir      =       config["RDIR"]   +   config["EXP"]   +   config["CYR"] + "-sp" +  str(config["ISP"]) + "/"
            os.makedirs(outdir, exist_ok=True)

            patterns = [
                config["RDIR"]   +   config["EXP"]   +   config["CVNREST"]  + config["CYR"] + "010100.bin-sp"     + str(config["ISP"]),
                config["RDIR"]   +   config["EXP"]   +   config["CVNREST"]  + config["CYR"] + "010100.bin.pth-sp" + str(config["ISP"]),
                config["RDIR"]   +   config["EXP"]   +   config["CVNREST"]  + config["CYR"] + "010100.nc-sp"      + str(config["ISP"]),
                config["RDIR"]   +   config["EXP"]   +   "**"     +   config["CYR"]      +    ".bin",
                config["RDIR"]   +   config["EXP"]   +   "**"     +   config["CYR"]      +    ".pth",
                config["RDIR"]   +   config["EXP"]   +   "**o_*"   +   config["CYR"]      +    ".nc",
                config["RDIR"]   +   config["EXP"]   +   "**"     +   config["CYR"] + ".log",
                config["RDIR"]   +   config["EXP"]   +   "log_CaMa-" + config["CYR"] + ".txt",
            ]

            for pattern in patterns:
                for file in glob.glob(pattern, recursive=True):
                    try:
                        shutil.move(file, os.path.join(outdir, os.path.basename(file)))
                    except Exception:
                        pass

            config["ISP"]       +=      1
        else:
            config["ISP"]        =      1
            config["IYR"]       +=      1

    else:
        config["IYR"]            +=     1

        # (6) Spin-up file management
        if config["IYR"] == config["YSTA"] and config["ISP"] <= config["NSP"]:
            config["IYR1"]    =     config["IYR"] + 1
            config["CYR1"]    =     config["IYR1"]

    # ================================================
    # (7) End of each year loop. Back to (3)

    return config