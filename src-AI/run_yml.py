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
from main_cmf import main_cmAI

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


def run_config(config: dict):
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

    while config  ["IYR"]   <=  config["YEND"]:
        config["CYR"]                   =       f"{config['IYR']:04d}"

        #   *** 3a. modify restart setting
        if config["SPINUP"] == 0:
            config["LRESTART"]          =       False             ## from zero storage
            config["CRESTSTO"]          =       ""
        else:
            config["LRESTART"]          =       True
            config["CRESTSTO"] = f"{config['CVNREST']}{config['CYR']}010100.bin"       ## from restart file
            if config["LRESTCDF"]:
                config["CRESTSTO"] = f"{config['CVNREST']}{config['CYR']}010100.nc"

        #   *** 3b. update start-end year
        config["SYEAR"]                 =       config["IYR"]
        config["SMON"]                  =       1
        config["SDAY"]                  =       1
        config["SHOUR"]                 =       0

        config["EYEAR"]                 =       config["SYEAR"]     +   1
        config["EMON"]                  =       1
        config["EDAY"]                  =       1
        config["EHOUR"]                 =       0

        # shutil.copy(config['PROG'] + config['EXE'], config['EXE'])

        # *** 3c. update input / output file data
        config["CSYEAR"]                =       f"{config['SYEAR']:04d}"
        config["COUTTAG"]               =       config['CSYEAR']
        config["CROFCDF"]               =       f"{config['CROFDIR']}{config['CROFPRE']}{config['CSYEAR']}{config['CROFSUF']}"
        config["SYEAR"]                 =       config["IYR"]
        config["SMONIN"]                =       1
        config["SDAYIN"]                =       1
        config["SHOURIN"]               =       0

        # ================================================
        # (4) Create NAMELIST for simulation year
        # it is OK to remove optional variables (set to default in CaMa-Flood)

        file_path = config["NMLIST"]
        if os.path.exists(file_path):
            os.remove(file_path)

        # *** 0. config
        namelist_content = f"""/ 
        &NRUNVER
        LADPSTP  = {config['LADPSTP']}                  ! true: use adaptive time step
        LPTHOUT  = {config['LPTHOUT']}                  ! true: activate bifurcation scheme
        LDAMOUT  = {config['LDAMOUT']}                  ! true: activate dam operation (under development)
        LRESTART = {config['LRESTART']}                 ! true: initial condition from restart file
        /
        &NDIMTIME
        CDIMINFO = "{config['CDIMINFO']}"               ! text file for dimention information
        DT       = {config['DT']}                       ! time step length (sec)
        IFRQ_INP = {config['IFRQ_INP']}                 ! input forcing update frequency (hour)
        /
        &NPARAM
        PMANRIV  = {config['PMANRIV']}                  ! manning coefficient river
        PMANFLD  = {config['PMANFLD']}                  ! manning coefficient floodplain
        PDSTMTH  = {config['PDSTMTH']}                  ! downstream distance at river mouth [m]
        PCADP    = {config['PCADP']}                    ! CFL coefficient
        /
        """

        with open(config["NMLIST"], "a") as f:
            f.write(namelist_content)

        # *** 1. time
        namelist_content = f"""/ 
        &NSIMTIME
        SYEAR   = {config['YSTA']}                      ! start year
        SMON    = {config['SMON']}                      !  month 
        SDAY    = {config['SDAY']}                      !  day 
        SHOUR   = {config['SHOUR']}                                    !  houe
        EYEAR   = {config['EYEAR']}                      ! end year
        EMON    = {config['EMON']}                      !  month 
        EDAY    = {config['EDAY']}                      !  day 
        EHOUR   = {config['EHOUR']}                                      !  hour
        /
        """

        with open(config["NMLIST"], "a") as f:
            f.write(namelist_content)

        # *** 2. map
        namelist_content = f"""/ 
        &NMAP
        LMAPCDF    =  {config['LMAPCDF']}                ! * true for netCDF map input
        CNEXTXY    = "{config['CNEXTXY']}"              ! river network nextxy
        CGRAREA    = "{config['CGRAREA']}"              ! catchment area
        CELEVTN    = "{config['CELEVTN']}"              ! bank top elevation
        CNXTDST    = "{config['CNXTDST']}"              ! distance to next outlet
        CRIVLEN    = "{config['CRIVLEN']}"              ! river channel length
        CFLDHGT    = "{config['CFLDHGT']}"              ! floodplain elevation profile
        CRIVWTH    = "{config['CRIVWTH']}"              ! channel width
        CRIVHGT    = "{config['CRIVHGT']}"              ! channel depth
        CRIVMAN    = "{config['CRIVMAN']}"              ! river manning coefficient
        CPTHOUT    = "{config['CPTHOUT']}"              ! bifurcation channel table
        /
        """

        with open(config["NMLIST"], "a") as f:
            f.write(namelist_content)

        # *** 3. restart
        namelist_content = f"""/ 
        &NRESTART
        CRESTSTO = "{config['CRESTSTO']}"               ! restart file
        CRESTDIR = "{config['CRESTDIR']}"               ! restart directory
        CVNREST  = "{config['CVNREST']}"                ! restart variable name
        LRESTCDF = {config['LRESTCDF']}                 ! * true for netCDF restart file (double precision)
        IFRQ_RST = {config['IFRQ_RST']}                 ! restart write frequency (1-24: hour, 0:end of run)
        /
        """

        with open(config["NMLIST"], "a") as f:
            f.write(namelist_content)

        # *** 4. forcing
        namelist_content = f"""/
        &NFORCE
        LINPCDF  = {config['LINPCDF']}                  ! true for netCDF runoff
        LINTERP  = {config['LINTERP']}                  ! true for runoff interpolation using input matrix
        CINPMAT  = "{config['CINPMAT']}"                ! input matrix file name
        DROFUNIT = {config['DROFUNIT']}                 ! runoff unit conversion
        CROFDIR  = "{config['CROFDIR']}"                ! runoff             input directory
        CROFPRE  = "{config['CROFPRE']}"                ! runoff             input prefix
        CROFSUF  = "{config['CROFSUF']}"                ! runoff             input suffix
        /
        """

        with open(config["NMLIST"], "a") as f:
            f.write(namelist_content)

        # *** 5. outputs
        namelist_content = f"""/
        &NOUTPUT
        COUTDIR  = "{config['COUTDIR']}"                ! OUTPUT DIRECTORY
        CVARSOUT = "{config['CVARSOUT']}"               ! Comma-separated list of output variables to save 
        COUTTAG  = "{config['COUTTAG']}"                ! Output Tag Name for each experiment
        LOUTVEC  = .FALSE                              ! TRUE FOR VECTORIAL OUTPUT, FALSE FOR NX,NY OUTPUT
        LOUTCDF  = {config['LOUTCDF']}                  ! * true for netcdf outptu false for binary
        NDLEVEL  = 0                                   ! * NETCDF DEFLATION LEVEL 
        IFRQ_OUT = {config['IFRQ_OUT']}                 ! output data write frequency (hour)
        /
        """

        with open(config["NMLIST"], "a") as f:
            f.write(namelist_content)
            f.flush()
            f.close()

        # ================================================
        # (5) Execute main program
        start_time = time.time()
        print("start:" + str(config["SYEAR"]) + str(datetime.now()))
        main_cmAI(config)
        print(f"Elapsed time: str({time.time() - start_time:.2f}) seconds")
        print("end:" +  str(config["SYEAR"]) +  str(datetime.now()))

        if   os.path.exists(config["LOGOUT"]):
            shutil.move(config["LOGOUT"], f"log_CaMa-{config['CYR']}.txt")

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

                try:
                    os.replace(config["CVNREST"] + config["CYR1"] + "010100.bin",
                               config["CVNREST"] + config["CYR"] + "010100.bin-sp" + str(config["ISP"]))
                except FileNotFoundError:
                    pass
                try:
                    os.replace(config["CVNREST"] + config["CYR1"] + "010100.bin",
                               config["CVNREST"] + config["CYR"] + "010100.bin-sp")
                except FileNotFoundError:
                    pass
                try:
                    os.replace(config["CVNREST"] + config["CYR1"] + "010100.bin.pth",
                               config["CVNREST"] + config["CYR"] + "010100.bin.pth-sp" + str(config["ISP"]))
                except FileNotFoundError:
                    pass
                try:
                    os.replace(config["CVNREST"] + config["CYR1"] + "010100.bin.pth",
                               config["CVNREST"] + config["CYR"] + "010100.bin.pth")
                except FileNotFoundError:
                    pass

                try:
                    os.replace(config["CVNREST"] + config["CYR1"] + "010100.nc",
                               config["CVNREST"] + config["CYR"] + "010100.nc-sp" + str(config["ISP"]))
                except FileNotFoundError:
                    pass
                try:
                    os.replace(config["CVNREST"] + config["CYR1"] + "010100.nc",
                               config["CVNREST"] + config["CYR"] + "010100.nc")
                except FileNotFoundError:
                    pass

                outdir      =       config["RDIR"]   +   config["EXP"]   +   config["CYR"] + "-sp" +  str(config["ISP"])
                os.makedirs(outdir, exist_ok=True)

                patterns = [
                    config["CVNREST"]  + config["CYR"] + "010100.bin-sp"     + str(config["ISP"]),
                    config["CVNREST"]  + config["CYR"] + "010100.bin.pth-sp" + str(config["ISP"]),
                    config["CVNREST"]  + config["CYR"] + "010100.nc-sp"      + str(config["ISP"]),
                    config["CYR"]      +    ".bin",
                    config["CYR"]      +    ".pth",
                    "o_*"   +   config["CYR"]      +    ".nc",
                    "*"     +   config["CYR"] + ".log",
                    "log_CaMa-" + config["CYR"] + ".txt",
                ]

                for pattern in patterns:
                    for file in glob.glob(pattern):
                        try:
                            if not os.path.exists(outdir):
                                os.makedirs(outdir, exist_ok=True)
                            shutil.move(file, os.path.join(outdir, os.path.basename(file)))
                        except FileNotFoundError:
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