#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  cmf_ctrl_restart_mod (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
-- CMF_RESTART_NMLIST : set restart configuration info from namelist
-- CMF_RESTART_INIT   : Read restart file
-- CMF_RESTART_WRITE  : Write restart file
"""
import  os
import re
from fortran_tensor_2D import Ftensor_2D
from netCDF4 import Dataset
import torch


os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

def Creat_Var (name, units, long_name,CC_NMLIST, NCID):
    var = NCID.createVariable(
        name, 'f8', ('lon', 'lat', 'time'),
        zlib=True, complevel=6, fill_value=CC_NMLIST.DMIS.cpu().numpy()
    )
    var.long_name = long_name
    var.units = units
    return NCID

class CMF_RESTART_NMLIST_MOD:
    def __init__(self):
        #     !*** NAMELIST/NOUTPUT/ from inputnam
        self.CRESTSTO               =           "restart"                   # input restart file name
        #!
        self.CRESTDIR               =           "./"                        # output restart file directory
        self.CVNREST                =            "restart"                  # output restart prefix
        self.LRESTCDF               =           False                       # true: netCDF restart file
        self.LRESTDBL               =           True                        # true: binary restart in double precision
        self.IFRQ_RST               =           0                           # 0: only at last time, (1,2,3,...,24) hourly restart, 30: monthly restart
        # --------------------------------------------------------------------------------------------------------------
    def CMF_RESTART_NMLIST(self, config, CC_NMLIST):
        """
        ! reed setting from namelist
        ! -- Called from CMF_DRV_NMLIST
        """
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        # --------------------------------------------------------------------------------------------------------------
        # !*** 1. open namelist
        with open(log_filename, 'a') as log_file:
            log_file.write("\n!---------------------!\n")
            log_file.write(f"CMF::RESTART_NMLIST: namelist OPEN in unit:    {CC_NMLIST.CSETFILE}\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        # !*** 2. default value
        self.CRESTSTO               =           config['CRESTSTO']      if 'CRESTSTO'  in config  else self.CRESTSTO
        #！
        self.CRESTDIR               =           config['CRESTDIR']      if 'CRESTDIR'  in config  else self.CRESTDIR
        self.CVNREST                =           config['CVNREST']       if 'CVNREST'   in config  else self.CVNREST
        self.LRESTCDF               =           config['LRESTCDF']      if 'LRESTCDF'  in config  else self.LRESTCDF
        self.LRESTDBL               =           config['LRESTDBL']      if 'LRESTDBL'  in config  else self.LRESTDBL
        self.IFRQ_RST               =           config['IFRQ_RST']      if 'IFRQ_RST'  in config  else self.IFRQ_RST
        # --------------------------------------------------------------------------------------------------------------
        # !*** 3. read namelist
        if CC_NMLIST.CSETFILE != "NONE":
            with open(CC_NMLIST.CSETFILE, 'r') as NSETFILE:
                NSETFILE.seek(0)
                NSIMTIME = {}
                for line in NSETFILE:
                    line = line.strip()
                    if "=" in line:
                        key, value = map(str.strip, line.split("=", 1))
                        NSIMTIME[key] = value
                self.CRESTSTO = re.findall(r"^(\S+)", NSIMTIME["CRESTSTO"], re.MULTILINE)[0]

            with open(log_filename, 'a') as log_file:
                log_file.write("=== NAMELIST, NRESTART  ===\n")
                log_file.write(f"CRESTSTO:              {self.CRESTSTO.strip()}\n")
                log_file.write(f"CRESTDIR:              {self.CRESTDIR.strip()}\n")
                log_file.write(f"CVNREST:               {self.CVNREST.strip()}\n")
                log_file.write(f"LRESTCDF:              {self.LRESTCDF}\n")
                log_file.write(f"LRESTDBL:              {self.LRESTDBL}\n")
                log_file.write(f"IFRQ_RST:              {self.IFRQ_RST}\n")
                log_file.flush()
                log_file.close()

            NSETFILE.close()
    # --------------------------------------------------------------------------------------------------------------
    def CMF_RESTART_INIT(self, config, CC_NMLIST, CC_VARS, device, CM_NMLIST, Datatype, CU):
        """
        ! read restart file
        ! -- call from CMF_DRV_INIT
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def READ_REST_CDF(config, CC_NMLIST, CC_VARS, device, CM_NMLIST, Datatype, CU):

            self.CFILE                      =           os.path.join(config['RDIR'] + config['EXP']  + self.CRESTSTO.replace('"', ''))
            log_filename                    =           config['RDIR'] + config['LOGOUT']

            with open(log_filename, 'a') as log_file:
                log_file.write(f"READ_REST: read restart netcdf:      {self.CFILE}\n")
                log_file.flush()
                log_file.close()


            NCFILE                          =           Dataset(self.CFILE, mode='r')
            P2TEMP_                         =           NCFILE.variables['rivsto'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
            P2TEMP                          =           Ftensor_2D(torch.tensor(P2TEMP_, device=device),start_row=1, start_col=1)
            CC_VARS.P2RIVSTO                =           CU.mapP2vecP(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

            P2TEMP_                         =           NCFILE.variables['fldsto'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
            P2TEMP                          =           Ftensor_2D(torch.tensor(P2TEMP_, device=device),start_row=1, start_col=1)
            CC_VARS.P2FLDSTO                =           CU.mapP2vecP(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

            if not CC_NMLIST.LSTOONLY:
                P2TEMP_                     =           NCFILE.variables['rivout_pre'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.D2RIVOUT_PRE        =           CU.mapP2vecD(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)
                CC_VARS.D2RIVOUT[:,:]       =           CC_VARS.D2RIVOUT_PRE.raw().clone()

                P2TEMP_                     =           NCFILE.variables['fldout_pre'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.D2FLDOUT_PRE        =           CU.mapP2vecD(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)
                CC_VARS.D2FLDOUT[:,:]       =           CC_VARS.D2FLDOUT_PRE.raw().clone()

                P2TEMP_                     =           NCFILE.variables['rivdph_pre'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.D2RIVDPH_PRE        =           CU.mapP2vecD(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

                P2TEMP_                     =           NCFILE.variables['fldsto_pre'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.D2FLDSTO_PRE        =           CU.mapP2vecD(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

            if CC_NMLIST.LGDWDLY:
                P2TEMP_                     =           NCFILE.variables['gdwsto'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.P2GDWSTO            =           CU.mapP2vecP(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

            if CC_NMLIST.LDAMOUT:       #!!! added
                P2TEMP_                     =           NCFILE.variables['damsto'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.P2DAMSTO            =           CU.mapP2vecP(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

            if CC_NMLIST.LLEVEE:  # !!! added
                P2TEMP_                     =           NCFILE.variables['levsto'][0:CC_NMLIST.NX, 0:CC_NMLIST.NY, 0]
                P2TEMP                      =           Ftensor_2D(torch.tensor(P2TEMP_, device=device), start_row=1, start_col=1)
                CC_VARS.P2LEVSTO            =           CU.mapP2vecP(P2TEMP, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)

            if CC_NMLIST.LPTHOUT and not CC_NMLIST.LSTOONLY:
                P1PTH_                      =           NCFILE.variables['pthflw_pre'][0:CM_NMLIST.NPTHOUT, 0:CM_NMLIST.NPTHLEV, 0]
                P1PTH                      =           Ftensor_2D(torch.tensor(P1PTH_, device=device), start_row=1, start_col=1)

                NP_Index                    =           torch.arange(1, CM_NMLIST.NPTHOUT + 1, device=device)

                ID_M                        =           ((CM_NMLIST.PTH_UPST[NP_Index] > 0) & (CM_NMLIST.PTH_DOWN[NP_Index] > 0) ).nonzero(as_tuple=True)[0]

                CC_VARS.D1PTHFLW_PRE[:,:]   =           torch.tensor(0, dtype=Datatype.JPRB, device=device)
                CC_VARS.D1PTHFLW_PRE[NP_Index[ID_M], :]\
                                            =           P1PTH[NP_Index[ID_M],:]
            NCFILE.close()

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        if self.LRESTCDF:
            READ_REST_CDF(config, CC_NMLIST, CC_VARS, device, CM_NMLIST, Datatype, CU)
        else:
            print(f"AttributeError: has no attribute 'READ_REST_BIN'")
            raise
    # --------------------------------------------------------------------------------------------------------------

    def CMF_RESTART_WRITE(self,log_filename, CT_NMLIST,CC_NMLIST, CM_NMLIST, CC_VARS, Datatype, CU, device, config):
        """
        ! write restart files
        ! -- called CMF_from DRV_ADVANCE
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def WRTE_BIN_MAP(P2VAR,TNAM,IREC,CM_NMLIST):
            import numpy as np
            #   !=================
            IREC                   =        IREC    +   1

            #   !! Double Precision Restart
            if self.LRESTDBL:
                if CM_NMLIST.REGIONTHIS == 1:
                    with open(TNAM, 'r+b') as f:
                        f.seek((IREC - 1) * 8 * CM_NMLIST.NX * CM_NMLIST.NY)
                        f.write(P2VAR.raw().numpy().astype(np.float64).tobytes(order='C'))

            #   !! Single Precision Restart
            else:
                print("The 'vecP2mapR' code in 107-th Line for cmf_ctrl_restart_mod.py is needed to improved")
                print("The 'vecP2mapR' code in 108-th Line for cmf_ctrl_restart_mod.py is needed to improved")
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def WRTE_REST_BIN(log_filename, CT_NMLIST,CC_NMLIST, CM_NMLIST, CC_VARS):
            import torch
            import numpy as np
            #   !*** set file nam
            CDATE               =       f"{CT_NMLIST.JYYYYMMDD:08d}{CT_NMLIST.JHOUR:02d}"

            CFILE               =       os.path.join(self.CRESTDIR, self.CVNREST + CDATE + CC_NMLIST.CSUFBIN)
            with open(log_filename, 'a') as log_file:
                log_file.write(f" WRTE_REST_BIN: restart file:   {CFILE}\n")
                log_file.flush()
                log_file.close()

            #   !*** write restart data (2D map)
            TMPNAM              =       CFILE

            if self.LRESTDBL:
                if CM_NMLIST.REGIONTHIS:
                    Data            =           torch.zeros((CM_NMLIST.NX, CM_NMLIST.NY), dtype=torch.float64)
                    with open(CFILE, 'wb') as f:
                        f.write(Data.numpy().tobytes(order='C'))
            else:
                if CM_NMLIST.REGIONTHIS:
                    Data            =           torch.zeros((CM_NMLIST.NX, CM_NMLIST.NY), dtype=torch.float32)
                    with open(CFILE, 'wb') as f:
                        f.write(Data.numpy().tobytes(order='C'))

            RIREC = 0
            P2TMP                   =           CC_VARS.P2RIVSTO
            WRTE_BIN_MAP        (P2TMP, TMPNAM, RIREC)
            P2TMP                   =           CC_VARS.P2FLDSTO
            WRTE_BIN_MAP        (P2TMP, TMPNAM, RIREC)
            #   !!================
            #   !! additional restart data for optional schemes (only write required vars)
            # if not self.LSTOONLY:
            P2TMP                   =           CC_VARS.D2RIVOUT_PRE
            WRTE_BIN_MAP        (P2TMP, TMPNAM, RIREC)
            P2TMP                   =           CC_VARS.D2FLDOUT_PRE
            WRTE_BIN_MAP        (P2TMP, TMPNAM, RIREC)
            P2TMP                   =           CC_VARS.D2RIVDPH_PRE
            WRTE_BIN_MAP        (P2TMP, TMPNAM, RIREC)
            P2TMP                   =           CC_VARS.D2FLDSTO_PRE
            WRTE_BIN_MAP        (P2TMP, TMPNAM, RIREC)
            # if self.LSTOONLY:
            # P2TMP                   =           CC_VARS.P2GDWSTO
            # # if self.LDAMOUT:
            # P2TMP                   =           CC_VARS.P2DAMSTO
            # # if self.LLEVEE:
            # P2TMP                   =           CC_VARS.P2LEVSTO
            f.close()

            if CC_NMLIST.LPTHOUT:
                CFILE               =           self.CRESTSTO    +   ".pth"
                with open(log_filename, 'a') as log_file:
                    log_file.write(f" READ_REST: read restart binary:    {CFILE}\n")
                    log_file.flush()
                    log_file.close()

                if self.LRESTDBL:
                    with open(CFILE, 'rb') as f:
                        # 只读第一条记录（即从 offset = 0 开始）
                        DATA        =           f.read(8 * CC_NMLIST.NPTHOUT * CM_NMLIST.NPTHLEV)
                        P1PTH       =           np.frombuffer(DATA, dtype=np.float64).reshape(CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV)
                        P1PTH[:,:]  =           CC_VARS.D1PTHFLW_PRE[:,:]
                else:
                    # 单精度读取（float32）
                    with open(CFILE, 'rb') as f:
                        data        =           f.read(4 * CC_NMLIST.NPTHOUT * CC_NMLIST.NPTHLEV)
                        R1PTH       =           np.frombuffer(data, dtype=np.float32).reshape(CC_NMLIST.NPTHOUT, CC_NMLIST.NPTHLEV)
                        R1PTH[:, :] =           CC_VARS.D1PTHFLW_PRE[:,:].to(dtype=torch.float32)
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def WRTE_REST_CDF(log_filename, CT_NMLIST, CC_NMLIST, CM_NMLIST, CC_VARS, Datatype, CU, device, config):
            import torch
            from netCDF4 import Dataset
            #   !================================================
            #   !*** 1. set file name & tim
            self.XTIME               =       ((CT_NMLIST.KMINNEXT - CT_NMLIST.KMINSTART) * 60).to(dtype=Datatype.JPRB)
            self.CTIME               =       f"seconds since {CT_NMLIST.ISYYYY:04d}-{CT_NMLIST.ISMM:02d}-{CT_NMLIST.ISDD:02d} {CT_NMLIST.ISHOUR:02d}:{CT_NMLIST.ISMIN:02d}"

            self.CDATE               =       f"{CT_NMLIST.JYYYYMMDD:08d}{CT_NMLIST.JHOUR:02d}"
            self.CFILE               =       os.path.join(config['RDIR'] + config['EXP']  + self.CVNREST + self.CDATE + CC_NMLIST.CSUFCDF)
            with open(log_filename, 'a') as log_file:
                log_file.write(f"WRTE_REST:create RESTART NETCDF:     {self.CFILE}\n")
                log_file.flush()
                log_file.close()

            #   !================================================
            #   !*** 2. create netCDF file
            # !! Note: all restart variables are saved as Float64.
            if CM_NMLIST.REGIONTHIS == 1:
                NCID            =       Dataset(self.CFILE, 'w', format='NETCDF4')

                #     !! dimensions
                NCID.createDimension('time', None)  # Unlimited dimension
                NCID.createDimension('lat', CC_NMLIST.NY)  # 固定维度
                NCID.createDimension('lon', CC_NMLIST.NX)

                if CC_NMLIST.LPTHOUT:
                    NCID.createDimension('NPTHOUT', CM_NMLIST.NPTHOUT)
                    NCID.createDimension('NPTHLEV', CM_NMLIST.NPTHLEV)

                #     !! dimensions
                lat              =      NCID.createVariable('lat', 'f4', ('lat',))
                lat.long_name    =      'latitude'
                lat.units        =      'degrees_north'

                lon              =      NCID.createVariable('lon', 'f4', ('lon',))
                lon.long_name    =      'longitude'
                lon.units        =      'degrees_east'

                time             =      NCID.createVariable('time', 'f8', ('time',))
                time.long_name   =      'time'
                time.units       =      self.CTIME

                #   !! variables
                NCID        =       Creat_Var('rivsto', 'm3', 'river storage', CC_NMLIST, NCID)
                NCID           =      Creat_Var('fldsto', 'm3', 'flood plain storage', CC_NMLIST, NCID)


                if not CC_NMLIST.LSTOONLY:      #   !! default restart with previous t-step outflw
                    NCID           =      Creat_Var   ('rivout_pre', 'm3/s', 'river outflow prev', CC_NMLIST, NCID)
                    NCID           =      Creat_Var   ('fldout_pre', 'm3/s', 'floodplain outflow prev', CC_NMLIST, NCID)
                    NCID           =      Creat_Var   ('rivdph_pre', 'm', 'river depth prev', CC_NMLIST, NCID)
                    NCID           =      Creat_Var   ('fldsto_pre', 'm3', 'floodplain storage prev', CC_NMLIST, NCID)

                    #       !! optional variables
                    if CC_NMLIST.LPTHOUT:
                        pthflw_pre  =       NCID.createVariable('pthflw_pre', 'f8', ('NPTHOUT', 'NPTHLEV', 'time'),
                                            zlib=True, complevel=6, fill_value=CC_NMLIST.DMIS.cpu().numpy())
                        pthflw_pre.long_name    =       "bifurcation outflow pre"
                        pthflw_pre.units        =       "m3/s"

                if CC_NMLIST.LGDWDLY:
                    print("The 'gdwsto' code in 255-th Line for main_cmf.py is needed to improved")
                    print("The 'gdwsto' code in 256-th Line for main_cmf.py is needed to improved")

                if CC_NMLIST.LDAMOUT:       #   !!! added
                    print("The 'damsto' code in 259-th Line for main_cmf.py is needed to improved")
                    print("The 'damsto' code in 260-th Line for main_cmf.py is needed to improved")

                if CC_NMLIST.LLEVEE:        #   !!! added
                    print("The 'levsto' code in 263-th Line for main_cmf.py is needed to improved")
                    print("The 'levsto' code in 264-th Line for main_cmf.py is needed to improved")

                #   !================================================
                #   !*** 2. write data
                #
                #   !! dimentions (time,lon,lat)
                NCID.variables['time'][:]       =       self.XTIME.cpu().numpy()
                NCID.variables['lon'][:]        =       CM_NMLIST.D1LON.raw().cpu().numpy()
                NCID.variables['lat'][:]        =       CM_NMLIST.D1LAT.raw().cpu().numpy()


            #   !! write restart variables (gather data in MPI mode)
            for JF in range(1, 10):
                IOUT = False
                CVAR = ''
                P2TEMP = None

                if JF == 1:
                    CVAR = 'rivsto'
                    P2TEMP = CU.vecP2mapP(CC_VARS.P2RIVSTO,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 2:
                    CVAR = 'fldsto'
                    P2TEMP = CU.vecP2mapP(CC_VARS.P2FLDSTO,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 3 and not CC_NMLIST.LSTOONLY:
                    CVAR = 'rivout_pre'
                    P2TEMP = CU.vecP2mapP(CC_VARS.D2RIVOUT_PRE,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 4 and not CC_NMLIST.LSTOONLY:
                    CVAR = 'fldout_pre'
                    P2TEMP = CU.vecP2mapP(CC_VARS.D2FLDOUT_PRE,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 5 and not CC_NMLIST.LSTOONLY:
                    CVAR = 'rivdph_pre'
                    P2TEMP = CU.vecP2mapP(CC_VARS.D2RIVDPH_PRE,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 6 and not CC_NMLIST.LSTOONLY:
                    CVAR = 'fldsto_pre'
                    P2TEMP = CU.vecP2mapP(CC_VARS.D2FLDSTO_PRE,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 7 and CC_NMLIST.LGDWDLY:
                    CVAR = 'gdwsto'
                    P2TEMP = CU.vecP2mapP(CC_VARS.P2GDWSTO,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 8 and CC_NMLIST.LDAMOUT:
                    CVAR = 'damsto'
                    P2TEMP = CU.vecP2mapP(CC_VARS.P2DAMSTO,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                elif JF == 9 and CC_NMLIST.LLEVEE:
                    CVAR = 'levsto'
                    P2TEMP = CU.vecP2mapP(CC_VARS.P2LEVSTO,CM_NMLIST.I1SEQX,CM_NMLIST.I1SEQY,CM_NMLIST.NSEQMAX,device)
                    IOUT = 1

                #   #ifdef UseMPI_CMF
                #   CALL CMF_MPI_AllReduce_P2MAP(P2TEMP)
                #   #endif
                if IOUT == 1 and CM_NMLIST.REGIONTHIS == 1:
                    try:
                        NCID.variables[CVAR][ :, :, 0] = P2TEMP.raw().cpu().numpy()
                    except KeyError:
                        print(f"Variable {CVAR} not found in NetCDF file.")

            if CC_NMLIST.LPTHOUT and not CC_NMLIST.LSTOONLY:
                P1PTH           =           torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV), dtype=Datatype.JPRD,device=device)
                P1PTH           =           Ftensor_2D(P1PTH, start_row=1, start_col=1)
                P1PTH[:,:]      =           CC_VARS.D1PTHFLW_PRE
                if CM_NMLIST.REGIONTHIS == 1:
                    try:
                        NCID.variables['pthflw_pre'][:, :, 0]    = P1PTH.raw().cpu().numpy()
                    except KeyError:
                        print("Variable 'pthflw_pre' not found in NetCDF.")

            with open(log_filename, 'a') as log_file:
                log_file.write("=== NAMELIST, NRESTART  ===\n")
                log_file.write(f"WRTE_REST: WRITE RESTART NETCDF:      {self.CFILE}\n")
                log_file.flush()
                log_file.close()
            NCID.close()  # !! regionthis=1: definition

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        IREST = 0

        if self.IFRQ_RST >= 0 and CT_NMLIST.KSTEP == CT_NMLIST.NSTEPS:      #   !! end of run
            IREST = 1

        elif 1 <= self.IFRQ_RST <= 24:                                      #   !! at selected hour
            if CT_NMLIST.JHOUR % self.IFRQ_RST == 0 and CT_NMLIST.JMIN == 0:
                IREST = 1

        elif self.IFRQ_RST == 30:
            if CT_NMLIST.JDD == 1 and CT_NMLIST.JHOUR == 0 and CT_NMLIST.JMIN == 0:     #   !! at start of month
                IREST = 1

        # === 写入重启文件 ===
        if IREST == 1:
            with open(log_filename, 'a') as log_file:
                log_file.write(f"\n!---------------------!\n")
                log_file.write(f"CMF::RESTART_WRITE: write time:   {CT_NMLIST.JYYYYMMDD} {CT_NMLIST.JHHMM}\n")
                log_file.flush()
                log_file.close()
            if self.LRESTCDF:           #   !! netCDF restart write
                WRTE_REST_CDF(log_filename, CT_NMLIST, CC_NMLIST, CM_NMLIST, CC_VARS, Datatype, CU, device, config)
            else:
                WRTE_REST_BIN(log_filename, CT_NMLIST,CC_NMLIST, CM_NMLIST, CC_VARS)
