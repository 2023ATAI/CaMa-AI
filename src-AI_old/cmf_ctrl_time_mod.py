#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  Manage time-related variables in CaMa-Flood (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_TIME_INIT   : Initialize    time-related variables
! -- CMF_TIME_NEXT   : Set next-step time-related variables
! -- CMF_TIME_UPDATE : Update        time-related variables
"""
import  os
import torch
import re


os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'



class CMF_CTRL_TIME_MOD:
    def __init__(self,   config,    Datatype):
        self.device         =       config["device"]
        # --------------------------------------------------------------------------------------------------------------
        #! simulation time step
        self.KSTEP_TyPe         = Datatype.JPIM                  #    time step since start
        self.NSTEPS_TyPe        = Datatype.JPIM                  #    total time step (from start to end), given in CMF_TIME_INIT
        # --------------------------------------------------------------------------------------------------------------
        # ! elapsed minute from base date (YYYY0,MM0,DD0)
        self.KMIN_TyPe          = Datatype.JPIM                  #    KMIN at the start of time step
        self.KMINNEXT_TyPe      = Datatype.JPIM                  #    KMIN at the end   of time step
        # --------------------------------------------------------------------------------------------------------------
        self.KMINSTART_TyPe     = Datatype.JPIM                  #    KMIN at the start of forcing runoff  data (netCDF)
        self.KMINSTASL_TyPe     = Datatype.JPIM                  #    KMIN at the start of boundary sealev data (netCDF)
        # --------------------------------------------------------------------------------------------------------------
        #! simulation start date:hour (KMINSTART)
        self.ISYYYYMMDD_TyPe    = Datatype.JPIM                  #    date     of simulation end time
        self.ISHHMM_TyPe        = Datatype.JPIM                  #    hour+min of simulation end time
        self.ISYYYY_TyPe        = Datatype.JPIM
        self.ISMM_TyPe          = Datatype.JPIM
        self.ISDD_TyPe          = Datatype.JPIM
        self.ISHOUR_TyPe        = Datatype.JPIM
        self.ISMIN_TyPe         = Datatype.JPIM
        # --------------------------------------------------------------------------------------------------------------
        #!simulation end   date:hour (KMINEND)
        self.IEYYYYMMDD_TyPe    = Datatype.JPIM                   #      date     of simulation end time
        self.IEHHMM_TyPe        = Datatype.JPIM                   #      hour+min of simulation end time
        self.IEYYYY_TyPe        = Datatype.JPIM
        self.IEMM_TyPe          = Datatype.JPIM
        self.IEDD_TyPe          = Datatype.JPIM
        self.IEHOUR_TyPe        = Datatype.JPIM
        self.IEMIN_TyPe         = Datatype.JPIM
        # --------------------------------------------------------------------------------------------------------------
        # !*** date:hour at START of time steop (KMIN)
        self.IYYYYMMDD_TyPe     = Datatype.JPIM                   #     date     at the start of time-step
        self.IYYYY_TyPe         = Datatype.JPIM                   #     year     at the start of time-step
        self.IMM_TyPe           = Datatype.JPIM                   #     month    at the start of time-step
        self.IDD_TyPe           = Datatype.JPIM                   #     day      at the start of time-step
        self.IHHMM_TyPe         = Datatype.JPIM                   #     hour+min at the start of time-step
        self.IHOUR_TyPe         = Datatype.JPIM                   #     hour     at the start of time-step
        self.IMIN_TyPe          = Datatype.JPIM                   #     min      at the start of time-step
        # --------------------------------------------------------------------------------------------------------------
        # !*** date:hour at START of time steop (KMIN)
        self.JYYYYMMDD_TyPe     = Datatype.JPIM                   #     date     at the end of time-step
        self.JYYYY_TyPe         = Datatype.JPIM                   #     year     at the end of time-step
        self.JMM_TyPe           = Datatype.JPIM                   #     month    at the end of time-step
        self.JDD_TyPe           = Datatype.JPIM                   #     day      at the end of time-step
        self.JHHMM_TyPe         = Datatype.JPIM                   #     hour+min at the end of time-step
        self.JHOUR_TyPe         = Datatype.JPIM                   #     hour     at the end of time-step
        self.JMIN_TyPe          = Datatype.JPIM                   #     min      at the end of time-step
        # --------------------------------------------------------------------------------------------------------------
        # *** base time to define kmin
        self.YYYY0_TyPe = Datatype.JPIM                      #     base year
        self.MM0_TyPe = Datatype.JPIM                        #     base month
        self.DD0_TyPe = Datatype.JPIM                        #     base day
        # --------------------------------------------------------------------------------------------------------------
        #!!=== NAMELIST/NSIMTIME/
        self.SYEAR_TyPe = Datatype.JPIM                      # START YEAR
        self.SMON_TyPe = Datatype.JPIM                       # START MONTH
        self.SDAY_TyPe = Datatype.JPIM                       # START DAY
        self.SHOUR_TyPe = Datatype.JPIM                      # START HOUR
        self.EYEAR_TyPe = Datatype.JPIM                      # END   YEAR
        self.EMON_TyPe = Datatype.JPIM                       # END   MONTH
        self.EDAY_TyPe = Datatype.JPIM                       # END   DAY
        self.EHOUR_TyPe = Datatype.JPIM                      # END   HOUR
        #NAMELIST/NSIMTIME/ SYEAR,SMON,SDAY,SHOUR, EYEAR,EMON,EDAY,EHOUR.
        #   !*** 1. set default value
        self.SYEAR      = torch.tensor    (2000,      dtype=self.SYEAR_TyPe,          device=self.device)
        self.SMON       = torch.tensor    (1,         dtype=self.SMON_TyPe,           device=self.device)
        self.SDAY       = torch.tensor    (1,         dtype=self.SDAY_TyPe,           device=self.device)
        self.SHOUR      = torch.tensor    (0,         dtype=self.SHOUR_TyPe,          device=self.device)
        self.EYEAR      = torch.tensor    (2001,      dtype=self.EYEAR_TyPe,          device=self.device)
        self.EMON       = torch.tensor    (1,         dtype=self.EMON_TyPe,           device=self.device)
        self.EDAY       = torch.tensor    (1,         dtype=self.EDAY_TyPe,           device=self.device)
        self.EHOUR      = torch.tensor    (0,         dtype=self.EHOUR_TyPe,          device=self.device)


        self.KMIN       = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)
        self.JHOUR      = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)
        self.JMIN       = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)
        self.JYYYY      = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)
        self.JMM        = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)
        self.JDD        = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)

        self.ISYYYY     = torch.tensor    (0,         dtype=self.SYEAR_TyPe,          device=self.device)
        self.test_cost  = {}

        # --------------------------------------------------------------------------------------------------------------
    def CMF_TIME_NMLIST(self, config, CC_NMLIST):
        """
        @purpose:  reed setting from namelist (python)
                    -- Called from CMF_DRV_NMLIST
        """
        # --------------------------------------------------------------------------------------------------------------
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write("\n!---------------------!\n")
            log_file.write(f"CMF::TIME_NMLIST: namelist OPEN in unit: {CC_NMLIST.CSETFILE}\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        self.SYEAR          =       torch.tensor    (config["SYEAR"]   if "SYEAR"  in config  else self.SYEAR,
                                                     dtype=self.SYEAR_TyPe,           device=self.device)
        self.SMON           =       torch.tensor    (config["SMON"]    if "SMON"  in config  else self.SMON,
                                                     dtype=self.SMON_TyPe ,           device=self.device)
        self.SDAY           =       torch.tensor    (config["SMON"]     if "SMON"  in config  else self.SDAY,
                                                     dtype=self.SDAY_TyPe ,           device=self.device)
        self.SHOUR          =       torch.tensor    (config["SHOUR"]   if "SHOUR"  in config  else self.SHOUR,
                                                     dtype=self.SHOUR_TyPe ,           device=self.device)
        self.EYEAR          =       torch.tensor   (config["EYEAR"]     if "EYEAR"  in config  else self.EYEAR,
                                                     dtype=self.EYEAR_TyPe ,           device=self.device)
        self.EMON           =       torch.tensor    (config["EMON"]     if "EMON"  in config  else self.EMON,
                                                     dtype=self.EMON_TyPe ,           device=self.device)
        self.EDAY           =       torch.tensor    (config["EDAY"]     if "EDAY"  in config  else self.EDAY,
                                                     dtype=self.EDAY_TyPe ,           device=self.device)
        self.SPINUP         =       config['SPINUP']

        self.NSP            =       config['NSP']
        # --------------------------------------------------------------------------------------------------------------
        # #!*** 2. read namelist
        # if CC_NMLIST.CSETFILE != "NONE":
        #     with open(CC_NMLIST.CSETFILE, 'r') as NSETFILE:
        #         NSETFILE.seek(0)
        #         NSIMTIME = {}
        #         for line in NSETFILE:
        #             line = line.strip()
        #             if "=" in line:
        #                 key, value = map(str.strip, line.split("=", 1))
        #                 NSIMTIME[key] = value
        #             # Extract required variables, keeping only numeric parts
        #         self.SYEAR      =                self.extract_numbers(NSIMTIME.get("SYEAR", "0"))
        #         self.SMON       =                self.extract_numbers(NSIMTIME.get("SMON",  "0"))
        #         self.SDAY       =                self.extract_numbers(NSIMTIME.get("SDAY",  "0"))
        #         self.SHOUR      =                self.extract_numbers(NSIMTIME.get("SHOUR", "0"))
        #         self.EYEAR      =                self.extract_numbers(NSIMTIME.get("EYEAR", "0"))
        #         self.EMON       =                self.extract_numbers(NSIMTIME.get("EMON",  "0"))
        #         self.EDAY       =                self.extract_numbers(NSIMTIME.get("EDAY",  "0"))
        #         self.EHOUR      =                self.extract_numbers(NSIMTIME.get("EHOUR", "0"))
        #     NSETFILE.close()

        # --------------------------------------------------------------------------------------------------------------
        #!*** 3. close namelist

        with open(log_filename, 'a') as log_file:
            log_file.write("=== NAMELIST, NSIMTIME ===\n")
            log_file.write(
                f"SYEAR, SMON, SDAY, SHOUR:              {self.SYEAR} {self.SMON} {self.SDAY} {self.SHOUR}\n")
            log_file.write(
                f"EYEAR, EMON, EDAY, EHOUR:              {self.EYEAR} {self.EMON} {self.EDAY} {self.EHOUR}\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        # Step 4: Define base date for KMIN calculation
        self.YYYY0           =          self.SYEAR.to(dtype=self.YYYY0_TyPe,              device=self.device)
        self.MM0             =          torch.tensor(1,              dtype=self.MM0_TyPe,                device=self.device)
        self.DD0             =          torch.tensor(1,              dtype=self.DD0_TyPe,                device=self.device)

        with open(log_filename, 'a') as log_file:
            log_file.write(f"TIME_NMLIST: YYYY0 MM0 DD0 set to:     {self.YYYY0} {self.MM0} {self.DD0}\n")

            log_file.write("CMF::TIME_NMLIST: end:")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
    # Function to extract numeric parts from a string
    def extract_numbers(self, value):
        numbers = re.findall(r'\d+', value)  # Find all digit sequences
        return int(numbers[0]) if numbers else None  # Return the first match or default



    def CMF_TIME_INIT       (self,              CC_NMLIST,        log_filename,        CU):
        """
        @purpose:  reed setting from namelist (python)
                    -- Called from CMF_DRV_NMLIST
        """

        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n!******************************!\n")
            log_file.write(f"CMF::DRV_INIT: initialization start\n")
            log_file.flush()
            log_file.close()

        #   !*** 1. Start time & End Time
        self.ISYYYYMMDD               =               self.SYEAR * 10000 + self.SMON * 100 + self.SDAY
        self.ISHHMM                   =               self.SHOUR * 100
        self.ISYYYY                   =               self.SYEAR
        self.ISMM                     =               self.SMON
        self.ISDD                     =               self.SDAY
        self.ISHOUR                   =               self.SHOUR
        self.ISMIN                    =               0


        self.IEYYYYMMDD               =               self.EYEAR * 10000 + self.EMON * 100 + self.EDAY  # End time
        self.IEHHMM                   =               self.EHOUR * 100
        self.IEYYYY                   =               self.EYEAR
        self.IEMM                     =               self.EMON
        self.IEDD                     =               self.EDAY
        self.IEHOUR                   =               self.EHOUR
        self.IEMIN                    =               0

        self.KMINSTART                = torch.tensor(0, dtype=self.KMINSTART_TyPe, device=self.device)
        self.KMIN                     = torch.tensor(0, dtype=self.KMIN_TyPe, device=self.device)


        #   !*** 2. Initialize KMIN for START & END Time
        self.MINSTART                 =               (CU.DATE2MIN
                                                      (self.ISYYYYMMDD,          self.ISHHMM,        self.YYYY0,
                                                      log_filename                                                      ))
        self.KMINEND                  =               (CU.DATE2MIN
                                                      (self.IEYYYYMMDD,          self.IEHHMM,        self.YYYY0,
                                                              log_filename                                              ))

        self.KMIN                     =                self.KMINSTART

        with open(log_filename, 'a') as log_file:
            log_file.write(f"Base Year YYYY0:   {self.YYYY0}\n")
            log_file.write(f"Start Date:        {self.ISYYYYMMDD,   self.ISHHMM,    self.KMINSTART}\n")
            log_file.write(f"End   Date:        {self.IEYYYYMMDD,   self.IEHHMM,    self.KMINEND}\n")
            log_file.flush()
            log_file.close()

        #   !*** 3. Calculate NSTEPS: time steps within simulation time
        self.KSTEP                    =           torch.tensor    (0.0,      dtype=self.KSTEP_TyPe,  device=self.device)
        NSTEPS_temp                   =           int(((self.KMINEND - self.KMINSTART) * 60) / CC_NMLIST.DT)  # (End - Start) / DT
        self.NSTEPS                   =           torch.tensor    (NSTEPS_temp,   dtype=self.NSTEPS_TyPe,  device=self.device)

        with open(log_filename, 'a') as log_file:
            log_file.write(f"NSTEPS:        {self.NSTEPS}\n")
            log_file.flush()
            log_file.close()

        #   !*** 4. Initial time step setting
        self.IYYYYMMDD                  =           self.ISYYYYMMDD
        self.IYYYY, self.IMM, self.IDD  =           CU.SPLITDATE(self.IYYYYMMDD)
        self.IHHMM                      =           self.ISHHMM
        self.IHOUR, self.IMIN           =           CU.SPLITHOUR(self.IHHMM)

        #   ! tentatively set KMINNEXT to KMIN (just within initialization phase)
        self.KMINNEXT                   =           self.KMIN
        self.JYYYYMMDD                  =           self.IYYYYMMDD
        self.JHHMM                      =           self.IHHMM
        self.JYYYY, self.JMM, self.JDD  =           CU.SPLITDATE(self.JYYYYMMDD)
        self.JHOUR, self.JMIN           =           CU.SPLITHOUR(self.JHHMM)

        with open(log_filename, 'a') as log_file:
            log_file.write(f"Initial Time Step Date:Hour : {self.IYYYYMMDD}  _  {self.IHOUR} : {self.IMIN}\n")
            #   !*** end
            log_file.write("CMF::TIME_INIT: end\n")
            log_file.flush()
            log_file.close()

        return CC_NMLIST

    def CMF_TIME_NEXT(self, CC_NMLIST, log_filename, CU, device):
        """
        @purpose:  ! update time-related valiable
                   ! -- Called from CMF_DRV_ADVANCE
        """
        #!*** 1. Advance KMIN, KSTEP
        self.KSTEP          =           self.KSTEP + 1
        self.KMINNEXT       =           self.KMIN + torch.tensor(int(CC_NMLIST.DT / 60),dtype=self.KMIN_TyPe, device=device)

        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write(f"\nCMF::TIME_NEXT:       {self.KSTEP}, {self.KMIN}, {self.KMINNEXT}, {CC_NMLIST.DT}\n")
            log_file.flush()
            log_file.close()

        self.JYYYYMMDD,self.JHHMM       =       CU.MIN2DATE(self.KMINNEXT, self.YYYY0, self.MM0, self.DD0)
        self.JYYYY,self.JMM,self.JDD    =       CU.SPLITDATE(self.JYYYYMMDD)
        self.JHOUR,self.JMIN            =       CU.SPLITHOUR(self.JHHMM)

        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write(f"\nStrt of Tstep: KMIN,     IYYYYMMDD, IHHMM       {self.KMIN}, {self.IYYYYMMDD}, {self.IHHMM}\n")
            log_file.write(f"\nEnd  of Tstep: KMINNEXT, JYYYYMMDD, JHHMM       {self.KMINNEXT}, {self.JYYYYMMDD}, {self.JHHMM}\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
    def CMF_TIME_UPDATE(self, log_filename):
        """
        @purpose:  ! update time-related valiable
                   ! -- Called from CMF_DRV_ADVANCE
        """
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\nCMF_TIME_UPDATE\n")
            log_file.flush()
            log_file.close()
        #   !*** 1. Advance KMIN, KSTEP
        self.KMIN = self.KMINNEXT

        # 2. Update I-time to J-time
        self.IYYYYMMDD = self.JYYYYMMDD
        self.IYYYY = self.JYYYY
        self.IMM = self.JMM
        self.IDD = self.JDD
        self.IHHMM = self.JHHMM
        self.IHOUR = self.JHOUR
        self.IMIN = self.JMIN

        with open(log_filename, 'a') as log_file:
            log_file.write(f"Current time update: KMIN={self.KMIN}, IYYYYMMDD={self.IYYYYMMDD}, IHHMM={self.IHHMM} \n")
            log_file.flush()
            log_file.close()
