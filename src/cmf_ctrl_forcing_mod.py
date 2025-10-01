#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@Co-author3: Cheng Zhang:  zc24@mails.jlu.edu.cn（Email）
@purpose:  Manage CaMa-Flood forcing (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_FORCING_NMLIST : Read setting from Namelist
! -- CMF_FORCING_INIT   : Initialize forcing data file
! -- CMF_FORCING_PUT    : Put forcing data (PBUFF) to CaMa-Flood
! -- CMF_FORCING_GET    : Read forcing data from file (save as "data buffer" PBUFF)
! -- CMF_FORCING_END    : Finalize forcing data file
"""
import  os
import re
import torch
import numpy as np
from fortran_tensor_2D import Ftensor_2D
from netCDF4 import Dataset

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


class RofCdf_Class:
    def __init__(self):
        self.CNAME: str = ""
        self.CVAR: list = [None, None, None]
        self.NVARID: list = [None, None, None]
        self.NSTART: int = 0

class CMF_FORCING_NMLIST_MOD:
    def __init__(self,  config,    Datatype,  CC_NMLIST):
        self.device                =                    config['device']
        #! Forcing configulation / foprcing mapping table "input matrix"
        self.LINPCDF                =                   False               # !! true : netCDF runoff forcing
        self.LINPEND                =                   False               # !! true  for input    endian conversion
        self.LINPDAY                =                   False               # !! true  for daily input file
        self.LINTERP                =                   False               # !! true : runoff interpolation using input matrix
        self.LITRPCDF               =                   False               # !! true : netCDF input matrix file
        self.CINPMAT                =                   "NONE"              # !! Input matrix filename
        self.DROFUNIT               =                   torch.tensor(86400*1000, dtype=Datatype.JPRB, device=self.device)
                                                                            # !! runoff unit conversion ( InpUnit/DROFUNIT = m3/m2/s)
        self.CROFDIR                =                   "./runoff/"         # !! Forcing: runoff directory
        self.CROFPRE                =                   "Roff____"          # !! Forcing: runoff prefix
        self.CROFSUF                =                   ".one"              # !! Forcing: runoff suffix
        #！
        self.CSUBDIR                =                   "./runoff/"         # !! Forcing: sub-surface runoff directory
        self.CSUBPRE                =                   "Rsub____"          # !! Forcing: sub-surface runoff prefix
        self.CSUBSUF                =                   ".one"              # !! Forcing: sub-surface runoff suffix
        #！  netCDF Forcing
        self.CROFCDF                =                   "NONE"              # !! Netcdf forcing file file
        self.CVNTIME                =                   "time"              # !! Netcdf forcing file file
        self.CVNROF                 =                   "runoff"            # !! Netcdf VARNAME of time dimention
        self.CVNSUB                 =                   "NONE"              # !! NetCDF VARNAME of sub-surface runoff.
        if  CC_NMLIST.LROSPLIT:
            self.CVNROF             =                   "Qs"
            self.CVNSUB             =                   "Qsb"

        self.SYEARIN                =                   torch.tensor(0, dtype=Datatype.JPIM, device=self.device)
                                                                             # !! START YEAR IN NETCDF INPUT RUNOFF
        self.SMONIN                 =                   torch.tensor(0, dtype=Datatype.JPIM, device=self.device)
                                                                             # !! START MONTH IN NETCDF INPUT RUNOFF
        self.SDAYIN                 =                   torch.tensor(0, dtype=Datatype.JPIM, device=self.device)
                                                                             # !! START DAY IN NETCDF INPUT RUNOFF
        self.SHOURIN                =                   torch.tensor(0, dtype=Datatype.JPIM, device=self.device)
                                                                             # !! START HOUR IN NETCDF INPUT RUNOFF
        # !* local variable
        self.NCID_TyPe              =                   Datatype.JPIM                # !! netCDF file     ID
        self.NVARID_2_TyPe          =                   Datatype.JPIM                # !! netCDF variable ID

        #! input matrix (converted from NX:NY*INPN to NSEQMAX*INPN)
        self.INPX_TyPe              =                   Datatype.JPIM                # !! INPUT GRID XIN
        self.INPY_TyPe              =                   Datatype.JPIM                # !! INPUT GRID YIN
        self.INPA_TyPe              =                   Datatype.JPRB                # !! INPUT AREA

        #! input matrix Inverse
        self.INPXI_TyPe              =                   Datatype.JPIM                # !! OUTPUT GRID XOUT
        self.INPYI_TyPe              =                   Datatype.JPIM                # !! OUTPUT GRID YOUT
        self.INPAI_TyPe              =                   Datatype.JPRB                # !! OUTPUT AREA
        self.INPNI_TyPe              =                   Datatype.JPIM                # !! MAX INPUT NUMBER for inverse interpolation

    def CMF_FORCING_NMLIST(self, config,  CT_NMLIST,  CC_NMLIST):
        """
        ! reed setting from namelist
        ! -- Called from CMF_DRV_NMLIST
        """
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        # --------------------------------------------------------------------------------------------------------------
        # !*** 1. open namelist
        with open(log_filename, 'a') as log_file:
            log_file.write("\n!---------------------!\n")
            log_file.write(f"CMF::FORCING_NMLIST: namelist OPEN in unit:   {CC_NMLIST.CSETFILE}\n")
            log_file.write("=== NAMELIST, NFORCE ===\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        # !*** 3. read namelist
        self.LINPCDF                =                   config['LINPCDF']      if 'LINPCDF'  in config  else self.LINPCDF
        self.LINPEND                =                   config['LINPEND']      if 'LINPEND'  in config  else self.LINPEND
        self.LINPDAY                =                   config['LINPDAY']      if 'LINPDAY'  in config  else self.LINPDAY
        self.LINTERP                =                   config['LINTERP']      if 'LINTERP'  in config  else self.LINTERP
        self.LITRPCDF               =                   config['LITRPCDF']     if 'LITRPCDF' in config  else self.LITRPCDF
        self.CINPMAT                =                   config['CINPMAT']      if 'CINPMAT' in config  else self.CINPMAT
        self.DROFUNIT               =                   config['DROFUNIT']     if 'DROFUNIT' in config  else self.DROFUNIT

        self.CROFDIR                =                   config['CROFDIR']      if 'CROFDIR'  in config  else self.CROFDIR
        self.CROFPRE                =                   config['CROFPRE']      if 'CROFPRE'  in config  else self.CROFPRE
        self.CROFSUF                =                   config['CROFSUF']      if 'CROFSUF'  in config  else self.CROFSUF
        #！
        self.CSUBDIR                =                   config['CSUBDIR']      if 'CSUBDIR'  in config  else self.CSUBDIR
        self.CSUBPRE                =                   config['CSUBPRE']      if 'CSUBPRE'  in config  else self.CSUBPRE
        self.CSUBSUF                =                   config['CSUBSUF']      if 'CSUBSUF'  in config  else self.CSUBSUF
        #！  netCDF Forcing
        self.CROFCDF                =                   config['CROFCDF']      if 'CROFCDF'  in config  else self.CROFCDF
        self.CVNTIME                =                   config['CVNTIME']      if 'CVNTIME'  in config  else self.CVNTIME
        self.CVNROF                 =                   config['CVNROF']       if 'CVNROF'   in config  else self.CVNROF
        self.CVNSUB                 =                   config['CVNSUB']       if 'CVNSUB'   in config  else self.CVNSUB

        self.SYEARIN                =                   config['SYEARIN']      if 'SYEARIN'  in config  else self.SYEARIN
        self.SMONIN                 =                   config['SMONIN']       if 'SMONIN'   in config  else self.SMONIN
        self.SDAYIN                 =                   config['SDAYIN']       if 'SDAYIN'   in config  else self.SDAYIN
        self.SHOURIN                =                   config['SHOURIN']      if 'SHOURIN'  in config  else self.SHOURIN

        # --------------------------------------------------------------------------------------------------------------
        if  CC_NMLIST.CSETFILE != "NONE":
            with open(CC_NMLIST.CSETFILE, 'r') as NSETFILE:
                NSETFILE.seek(0)
                NSIMTIME = {}
                for line in NSETFILE:
                    line = line.strip()
                    if "=" in line:
                        key, value = map(str.strip, line.split("=", 1))
                        NSIMTIME[key] = value
                # Extract required variables, keeping only numeric parts
                LINPCDF_val         =           re.findall(r"^(\S+)", NSIMTIME["LINPCDF"], re.MULTILINE)[0].strip('.')
                self.LINPCDF        =           False if LINPCDF_val in ['.FALSE.', 'FALSE'] else True
                LINTERP_val         =           re.findall(r"^(\S+)", NSIMTIME["LINTERP"], re.MULTILINE)[0].strip('.')
                self.LINTERP        =           False if LINTERP_val in ['.FALSE.', 'FALSE'] else True
                self.CINPMAT        =           re.findall(r"^(\S+)", NSIMTIME["CINPMAT"], re.MULTILINE)[0].strip('"')
                self.CROFDIR        =           re.findall(r"^(\S+)", NSIMTIME["CROFDIR"], re.MULTILINE)[0].strip('"')
                self.CROFPRE        =           re.findall(r"^(\S+)", NSIMTIME["CROFPRE"], re.MULTILINE)[0].strip('"')
            NSETFILE.close()

        if not CC_NMLIST.LINPCDF:
            self.LINPDAY            =                   True        #   ! for plain binary input, only DAILY runoff file accepted (v4.2)

        with open(log_filename, 'a') as log_file:
            log_file.write(f"LINPCDF                {self.LINPCDF}\n")
            log_file.write(f"LINPDAY                {self.LINPDAY}\n")
            log_file.write(f"LINTERP                {self.LINTERP}\n")
            log_file.write(f"LITRPCDF               {self.LITRPCDF}\n")
            log_file.write(f"CINPMAT                {self.CINPMAT.strip()}\n")
            log_file.write(f"LROSPLIT               {CC_NMLIST.LROSPLIT}\n")
            if self.LINPDAY:
                log_file.write(f"CROFDIR            {self.CROFDIR.strip()}\n")
                log_file.write(f"CROFPRE            {self.CROFPRE.strip()}\n")
                log_file.write(f"CROFSUF            {self.CROFSUF.strip()}\n")
            if not self.LINPCDF:                                        #   !! plain binary
                if CC_NMLIST.LROSPLIT:
                    log_file.write(f"CROFDIR            {self.CROFDIR.strip()}\n")
                    log_file.write(f"CROFPRE            {self.CROFPRE.strip()}\n")
                    log_file.write(f"CROFSUF            {self.CROFSUF.strip()}\n")
            else:
                if not self.LINPDAY:
                    log_file.write(f"CROFCDF                {self.CROFCDF.strip()}\n")
                    log_file.write(f"SYEARIN, SMONIN, SDAYIN, SHOURIN:          {self.SYEARIN, self.SMONIN, self.SDAYIN, self.SHOURIN:}\n")
                log_file.write(f"CVNTIME                    {self.CVNTIME.strip()}\n")
                log_file.write(f"CVNROF                     {self.CVNROF.strip()}\n")
                if CC_NMLIST.LROSPLIT:
                    log_file.write(f"CVNSUB             {self.CVNSUB.strip()}\n")
            if self.LINPEND:
                log_file.write(f"LINPEND         {self.LINPEND.strip()}\n")
        # --------------------------------------------------------------------------------------------------------------
        # !*** 4. modify base date (shared for KMIN)
        if not self.LINPDAY:
            if self.SYEARIN > 0:
                CT_NMLIST.YYYY0        =           torch.minimum(CT_NMLIST.YYYY0,torch.tensor(self.SYEARIN))
        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::FORCING_NMLIST: end\n")
            log_file.flush()
            log_file.close()
        return CT_NMLIST
        # --------------------------------------------------------------------------------------------------------------
    def CMF_FORCING_INIT(self, CC_NMLIST,   CT_NMLIST,   CU,     log_filename,    CM_NMLIST,     Datatype):
        """
        ! Initialize/open netcdf input
        ! -- called from "Main Program / Coupler"
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CMF_INPMAT_INIT_BIN(CC_NMLIST,log_filename,CM_NMLIST,C_U,device,Datatype):
            with open(log_filename, 'a') as log_file:
                log_file.write( f"NX, NY, INPN =      {CC_NMLIST.NX}   {CC_NMLIST.NY}   {CC_NMLIST.INPN}\n")
                log_file.write(f"INPUT MATRIX binary  {self.CINPMAT}\n")
                log_file.flush()
                log_file.close()

            self.INPX            =        torch.zeros((CM_NMLIST.NSEQMAX, CC_NMLIST.INPN), dtype=self.INPX_TyPe, device=device)
            self.INPX            =        Ftensor_2D(self.INPX, start_row=1, start_col=1)
            self.INPY            =        torch.zeros((CM_NMLIST.NSEQMAX, CC_NMLIST.INPN), dtype=self.INPY_TyPe, device=device)
            self.INPY            =        Ftensor_2D(self.INPY, start_row=1, start_col=1)
            self.INPA            =        torch.zeros((CM_NMLIST.NSEQMAX, CC_NMLIST.INPN), dtype=self.INPA_TyPe, device=device)
            self.INPA            =        Ftensor_2D(self.INPA, start_row=1, start_col=1)

            #   2. Read Input Matrix

            self.I2TMP            =       torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY), dtype=Datatype.JPIM, device=device)
            self.I2TMP            =       Ftensor_2D(self.I2TMP, start_row=1, start_col=1)
            self.R2TMP            =       torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY), dtype=Datatype.JPRM, device=device)
            self.R2TMP            =       Ftensor_2D(self.R2TMP, start_row=1, start_col=1)

            with open(self.CINPMAT, 'rb') as f:
                buffer            =       f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY * 3 * CC_NMLIST.INPN)
                data_np           =       np.frombuffer(buffer[:2 * CC_NMLIST.INPN * CC_NMLIST.NY * CC_NMLIST.NX* 4], dtype=np.int32)
                data_np_          =       np.frombuffer(buffer[2 * CC_NMLIST.INPN * CC_NMLIST.NY * CC_NMLIST.NX * 4:],dtype=np.float32)

                all_data          =       data_np.reshape((2 * CC_NMLIST.INPN, CC_NMLIST.NY, CC_NMLIST.NX)).transpose(0, 2, 1)
                all_data_         =       data_np_.reshape((1 * CC_NMLIST.INPN, CC_NMLIST.NY, CC_NMLIST.NX)).transpose(0, 2, 1)
                all_data          =       torch.tensor(all_data, device=device)
                all_data_         =       torch.tensor(all_data_, device=device)
            for i_ in range (CC_NMLIST.INPN):
                INPI              =        i_ + 1
                I2TMP_X           =        all_data[i_]  # 形状 [NX, NY]
                I2TMP_Y           =        all_data[CC_NMLIST.INPN + i_]
                R2TMP             =        all_data_[i_]
                self.INPX[:, INPI]      =           C_U.mapI2vecI_(Ftensor_2D(I2TMP_X, start_row=1, start_col=1),
                                                            CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)
                self.INPY[:, INPI]      =           C_U.mapI2vecI_(Ftensor_2D(I2TMP_Y, start_row=1, start_col=1),
                                                            CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)
                self.INPA[:, INPI]      =           C_U.mapR2vecD_(Ftensor_2D(R2TMP, start_row=1, start_col=1),
                                                            CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)
            return
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CMF_FORCING_INIT_CDF(CC_NMLIST, CT_NMLIST, log_filename, CU):
            if not self.LINPDAY:
                # --------------------------------------------------------------------------------------------------------------
                #   !*** 1. calculate KMINSTAINP (start KMIN for forcing)
                KMINSTAIN = (CU.DATE2MIN
                             (CT_NMLIST.IYYYY * 10000 + CT_NMLIST.IMM * 100 + CT_NMLIST.IDD, 0,
                              CT_NMLIST.YYYY0, log_filename))

                # 2. Initialize Type for Runoff CDF:


                self.ROFCDF                  =       RofCdf_Class()
                self.ROFCDF.CNAME            =       str(self.CROFCDF)
                self.ROFCDF.CVAR[0]          =       self.CVNROF
                self.ROFCDF.CVAR[1]          =       self.CVNSUB
                if not CC_NMLIST.LROSPLIT:
                    self.ROFCDF.CVAR[1]      =       "NONE"
                    self.ROFCDF.NVARID[1]    =       -1
                if not CC_NMLIST.LWEVAP:
                    self.ROFCDF.CVAR[2]      =       "NONE"
                    self.ROFCDF.NVARID[2]    =       -1

                self.ROFCDF.NSTART           =       KMINSTAIN
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CMF::FORCING_INIT_CDF: {self.ROFCDF.CNAME}, {KMINSTAIN}\n")
                    log_file.flush()
                    log_file.close()

                #     !*** 3. Open netCDF ruoff file
                NC_FILE = Dataset(self.ROFCDF.CNAME, mode='r')
                self.ROFCDF.NVARID[0] = NC_FILE.variables[self.ROFCDF.CVAR[0]]

                # CC_NMLIST.NXIN        =     self.ROFCDF.NVARID[0].shape[1]
                # CC_NMLIST.NYIN        =     self.ROFCDF.NVARID[0].shape[2]

                if CC_NMLIST.LROSPLIT:
                    self.ROFCDF.NVARID[1] = NC_FILE.variables[self.ROFCDF.CVAR[1]]
                if CC_NMLIST.LWEVAP:
                    self.ROFCDF.NVARID[2] = (NC_FILE.variables[self.ROFCDF.CVAR[2]])

                try:
                    TIME_DIM = NC_FILE.dimensions[self.CVNTIME]
                except KeyError:
                    raise RuntimeError(f"GETTING TIME ID FORCING RUNOFF: Time dimension not found: {self.CVNTIME}")
                NCDFSTP = len(TIME_DIM)

                with open(log_filename, 'a') as log_file:
                    log_file.write(
                        f"CMF::FORCING_INIT_CDF: CNAME, NCID, VARID = {self.ROFCDF.CNAME}, 'not set NCID and in pytorch' \n")
                    log_file.flush()
                    log_file.close()

                #   !*** 4. check runoff forcing time
                if CT_NMLIST.KMINSTART < KMINSTAIN:
                    with open(log_filename, 'a') as log_file:
                        log_file.write(
                            f"Run start earlier than forcing data, {self.ROFCDF.CNAME}, {CT_NMLIST.KMINSTART}, {KMINSTAIN} \n")
                        log_file.flush()
                        log_file.close()
                    raise RuntimeError(
                        f"Run start earlier than forcing data, {self.ROFCDF.CNAME}, {CT_NMLIST.KMINSTART}, {KMINSTAIN}")
                KMINENDIN = KMINSTAIN + NCDFSTP * int(CC_NMLIST.DTIN / 60)

                if CT_NMLIST.KMINEND > KMINENDIN:
                    with open(log_filename, 'a') as log_file:
                        log_file.write(
                            f"Run end later than forcing data, {self.ROFCDF.CNAME}, {CT_NMLIST.KMINEND}, {KMINENDIN} \n")
                        log_file.flush()
                        log_file.close()
                    raise RuntimeError(
                        f"Run end later than forcing data, {self.ROFCDF.CNAME}, {CT_NMLIST.KMINEND}, {KMINENDIN}")
            return CC_NMLIST
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write("\n!---------------------!\n")
            log_file.write(f"CMF::FORCING_INIT: Initialize runoff forcing file (only for netCDF)\n")
            log_file.flush()
            log_file.close()

        if self.LINPCDF:
            CC_NMLIST   =   CMF_FORCING_INIT_CDF(CC_NMLIST, CT_NMLIST, log_filename, CU)

        if self.LINTERP:
            if self.LITRPCDF:
                print("The 'CMF_INPMAT_INIT_CDF' code in 380-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
                print("The 'CMF_INPMAT_INIT_CDF' code in 381-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
            else:
                CMF_INPMAT_INIT_BIN(CC_NMLIST,log_filename,CM_NMLIST, CU, self.device, Datatype)

        with open(log_filename, 'a') as log_file:
            log_file.write("CMF::FORCING_INIT: end\n")
            log_file.flush()
            log_file.close()
        return  CC_NMLIST
    def CMF_FORCING_GET(self, CC_NMLIST,      CT_NMLIST,    Datatype,      PBUFF,      CU,     config):
        """
        ! read runoff from file
        """
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CMF_FORCING_GET_BIN(CC_NMLIST, PBUFF, CT_NMLIST, log_filename,device,Datatype):
        # --------------------------------------------------------------------------------------------------------------
            R2TMP           =           torch.zeros((CC_NMLIST.NXIN, CC_NMLIST.NYIN), dtype=Datatype.JPRM, device=device)
            R2TMP           =           Ftensor_2D(R2TMP, start_row=1, start_col=1)
            # *** 1. calculate IREC for sub-daily runoff
            ISEC            =           CT_NMLIST.IHOUR * 3600 + CT_NMLIST.IMIN * 60        # !! current second in a day
            self.IRECINP         =           (ISEC // CC_NMLIST.DTIN) + 1                        # !! runoff irec (sub-dairy runoff)

             # !*** 2. set file name

            CDATE           =           f"{CT_NMLIST.IYYYY:04d}{CT_NMLIST.IMM:02d}{CT_NMLIST.IDD:02d}"
            self.CIFNAME         =           f"{self.CROFDIR}/{self.CROFPRE}{CDATE}{self.CROFSUF}"
            with open(log_filename, 'a') as log_file:
                log_file.write(f"CMF::FORCING_GET_BIN: {self.CIFNAME}\n")
                log_file.flush()
                log_file.close()

            # !*** 3. open & read runoff
            with open(self.CIFNAME, 'rb') as f:
                buffer          =           f.read(4 * CC_NMLIST.NXIN * CC_NMLIST.NYIN)
                data_np         =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NYIN, CC_NMLIST.NXIN).T
                R2TMP[:, :]     =           torch.tensor(data_np, dtype=R2TMP.raw().dtype, device=device)
                f.close()
            with open(log_filename, 'a') as log_file:
                log_file.write(f"IRECINP: {self.IRECINP}\n")
                log_file.flush()
                log_file.close()

            # !*** 4. copy runoff to PBUSS, endian conversion is needed
            if self.LINPEND:
                print("The 'CONV_END' code in 318-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
                print("The 'CONV_END' code in 319-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
            PBUFF[:, :, 1]      =           R2TMP[:,:]
            # !*** for sub-surface runoff withe LROSPLIT
            PBUFF[:, :, 2]       =          torch.tensor(0, dtype=Datatype.JPRB, device=device)     #!! Plain Binary subsurface runoff to be added later
            if CC_NMLIST.LROSPLIT:
                self.CIFNAME = f"{self.CSUBDIR}/{self.CSUBPRE}{CDATE}{self.CSUBSUF}"
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CMF::FORCING_GET_BIN: (sub-surface): {self.CIFNAME}\n")
                    log_file.flush()
                    log_file.close()
                with open(self.CIFNAME, 'rb') as f:
                    buffer = f.read(4 * CC_NMLIST.NXIN * CC_NMLIST.NYIN)
                    data_np = np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NYIN, CC_NMLIST.NXIN).T
                    R2TMP[:, :] = torch.tensor(data_np, dtype=R2TMP.raw().dtype, device=device)
                    f.close()
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"IRECINP: {R2TMP.raw().ndim}\n")
                    log_file.flush()
                    log_file.close()

                if self.LINPEND:
                    print("The 'CONV_END' code in 340-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
                    print("The 'CONV_END' code in 341-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
                PBUFF[:, :, 2] = R2TMP[:, :]

            return PBUFF
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CMF_FORCING_GET_CDF(CC_NMLIST, CU, CT_NMLIST, log_filename, device):
        # --------------------------------------------------------------------------------------------------------------

            if self.LINPDAY:
                # *** 1. calculate IREC for sub-daily runoff
                ISEC            =           CT_NMLIST.IHOUR * 3600 + CT_NMLIST.IMIN * 60  # !! current second in a day
                self.IRECINP         =           (ISEC // CC_NMLIST.DTIN) + 1  # !! runoff irec (sub-dairy runoff)

                #   !! netCDF runoff file start time
                KMINSTA         =           (CU.DATE2MIN
                                            (CT_NMLIST.IYYYY * 10000 + CT_NMLIST.IMM * 100 + CT_NMLIST.IDD, 0,
                                            CT_NMLIST.YYYY0, log_filename))

                # !*** 2. set file name
                CDATE = f"{CT_NMLIST.IYYYY:04d}{CT_NMLIST.IMM:02d}{CT_NMLIST.IDD:02d}"
                self.CIFNAME = f"{self.CROFDIR}/{self.CROFPRE}{CDATE}{self.CROFSUF}"
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CMF::FORCING_GET_CDF: {self.CIFNAME}, {KMINSTA}\n")
                    log_file.flush()
                    log_file.close()



                self.ROFCDF.CNAME    =   self.CIFNAME
                self.ROFCDF.CVAR     =   [self.CVNROF, self.CVNSUB]
                if not CC_NMLIST.LROSPLIT:
                    self.ROFCDF.CVAR[1] = "NONE"
                    self.ROFCDF.NVARID[1] = [None, -1]
                if not CC_NMLIST.LWEVAP:
                    self.ROFCDF.CVAR[2] = "NONE"
                    self.ROFCDF.NVARID[2] = [None, None, -1]

                self.ROFCDF.NSTART = KMINSTA
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CMF::FORCING_GET_CDF: {self.CIFNAME}, {KMINSTA}\n")
                    log_file.flush()
                    log_file.close()

                #     !*** 3. Open netCDF ruoff file
                NC_FILE         =    Dataset(self.ROFCDF.CNAME, mode='r')

                var1            =    NC_FILE.variables[self.ROFCDF.CVAR[0]]
                if CC_NMLIST.LROSPLIT:
                    var2 = NC_FILE.variables[self.ROFCDF.CVAR[1]]

                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CMF::FORCING_GET_CDF: CNAME,NCID,VARID: {self.CIFNAME}, {NC_FILE}, {self.ROFCDF.NVARID[0]}\n")
                    log_file.flush()
                    log_file.close()

                #      !*** 4. read runoff
                # need modify the following code....
                # ..................................
            else:       #   !! LINPDAY=.false. : one runoff input file during simulation period
                #   !*** 1. calculate irec
                self.IRECINP = int((CT_NMLIST.KMIN - self.ROFCDF.NSTART) * 60 / CC_NMLIST.DTIN)      #   ! (second from netcdf start time) / (input time step)

                #    !*** 2. read runoff
                PBUFF[:, :, 1] = torch.tensor(self.ROFCDF.NVARID[0][self.IRECINP, :, :].T, device=device)

                if self.ROFCDF.NVARID[1] != -1:
                    PBUFF[:, :, 2]      =       torch.tensor(self.ROFCDF.NVARID[1][self.IRECINP, :, :].T, device=device)


                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CMF::FORCING_GET_CDF: read runoff: {CT_NMLIST.IYYYYMMDD}, {CT_NMLIST.IHHMM}, {self.IRECINP}\n")
                    log_file.flush()
                    log_file.close()

                return PBUFF
        # --------------------------------------------------------------------------------------------------------------

        if self.LINPCDF :
            PBUFF    =   CMF_FORCING_GET_CDF(CC_NMLIST, CU, CT_NMLIST, log_filename,self.device)
        else:
            PBUFF    =   CMF_FORCING_GET_BIN(CC_NMLIST, PBUFF, CT_NMLIST, log_filename,self.device,Datatype)

        #   !! Check if PRUFINN(IX,IY) is NaN (Not-A-Number) ot not
        PBUFF[:, :, 1]  =   torch.nan_to_num(PBUFF[:, :, 1], nan=CC_NMLIST.RMIS)
        PBUFF[:, :, 1]  =   torch.clamp(PBUFF[:, :, 1], min=0.0)

        if CC_NMLIST.LROSPLIT:
            PBUFF[:, :, 2] = torch.nan_to_num(PBUFF[:, :, 2], nan=CC_NMLIST.RMIS)
            PBUFF[:, :, 2] = torch.clamp(PBUFF[:, :, 2], min=0.0)
        return PBUFF
    def CMF_FORCING_PUT(self,  CC_NMLIST,      CM_NMLIST,     PBUFF,    config,     Datatype,   CC_VAR):
        """
        ! interporlate with inpmat, then send runoff data to CaMa-Flood
        ! -- called from "Main Program / Coupler" or CMF_DRV_ADVANCE
        """
        log_filename = config['RDIR'] + config['LOGOUT']
        # ==========================================================
        # + ROFF_INTERP : runoff interpolation with mass conservation using "input matrix table (inpmat)"
        # + CONV_RESOL : nearest point runoff interpolation
        # ==========================================================
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def ROFF_INTERP(CC_NMLIST, CM_NMLIST, PBUFFIN ,log_filename, device, Datatype):
            """
            ! interporlate runoff using "input matrix"
            Note:   Distribute runoff on the two-dimensional grid (in mm/dt) to river network cells according to specified
                    interpolation weights, convert it to volumetric flow rate (m³/s), and ensure mass conservation.
            """
            PBUFFIN         =       Ftensor_2D(PBUFFIN, start_row=1, start_col=1)
            PBUFFOUT        =       torch.zeros((CM_NMLIST.NSEQALL, 1), dtype=Datatype.JPRB, device=device)
            PBUFFOUT        =       Ftensor_2D(PBUFFOUT, start_row=1, start_col=1)

            # NQ_Index = torch.arange(1, CM_NMLIST.NSEQALL + 1, device=device)
            # IP_Index = torch.arange(1, CM_NMLIST.INPN + 1, device=device)
            NQ_Index, IP_Index = torch.meshgrid(
                torch.arange(1, CM_NMLIST.NSEQALL + 1, device=device),
                torch.arange(1, CC_NMLIST.INPN + 1, device=device),
                indexing='ij'
            )

            IXIN            =       self.INPX[NQ_Index, IP_Index]
            IYIN            =       self.INPY[NQ_Index, IP_Index]

            cond_pos_1      =       IXIN > 0
            cond_pos_2      =       (IXIN <= CC_NMLIST.NXIN) & (IYIN <= CC_NMLIST.NYIN)
            cond_out_2      =       (IXIN > CC_NMLIST.NXIN) | (IYIN > CC_NMLIST.NYIN)

            #computer valid PBUFFIN-------------------------------------------------------------------------------------
            cond_valid      =       cond_pos_1 & cond_pos_2

            IXIN_valid      =       IXIN[cond_valid]
            IYIN_valid      =       IYIN[cond_valid]
            NQ_valid        =       NQ_Index[cond_valid]
            IP_valid        =       IP_Index[cond_valid]

            valid_mask      =       (PBUFFIN[IXIN_valid, IYIN_valid] != CC_NMLIST.RMIS)

            IXIN_final      =       IXIN_valid[valid_mask]
            IYIN_final      =       IYIN_valid[valid_mask]
            NQ_final        =       NQ_valid[valid_mask] - 1  # index_add_ 从 0 开始
            IP_final        =       IP_valid[valid_mask]

            PBUFFIN_R       =       PBUFFIN[IXIN_final, IYIN_final]
            INPA_R          =       self.INPA[NQ_final + 1, IP_final]

            PBUFFOUT.raw().index_add_(
                0,
                NQ_final,
                (PBUFFIN_R * INPA_R / self.DROFUNIT).unsqueeze(1)
            )
            # ----------------------------------------------------------------------------------------------------------

            ID_D_M = (cond_pos_1 & cond_out_2).nonzero(as_tuple=True)[0]

            if ID_D_M.any():
                ISEQ_bad            =           NQ_Index[ID_D_M]
                INPI_bad            =           IP_Index[ID_D_M]
                IXIN_bad            =           self.INPX.raw()[ID_D_M]
                IYIN_bad            =           self.INPY.raw()[ID_D_M]
                for i in range(len(ISEQ_bad)):
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"error")
                        log_file.write(f"XXX  {ISEQ_bad[i]} {INPI_bad[i]} {IXIN_bad[i]} {IYIN_bad[i]}\n")
                        log_file.flush()
                        log_file.close()


            ISEQPtotal =  [1, 1, 1, 1, 70001,70001, 70001, 70001, 90001, 90001, 90001, 90001, 130001, 130001, 130001, 130001]
            GridpPtotal = [1, 2, 3, 4, 1,    2,     3,     4,      1,    2,     3,     4,     1,      2,      3,      4]
            for ii in range(len(ISEQPtotal)):
                ISEQP = ISEQPtotal[ii]
                Gridp = GridpPtotal[ii]
                INPXp = self.INPX[ISEQP, Gridp]
                INPYp = self.INPY[ISEQP, Gridp]
                INPAp = self.INPA[ISEQP, Gridp]
                print(f'river point                 {ISEQP}')
                print(f'Valid contributing grids    {Gridp}')
                print(f'INPX=                       {INPXp}')
                print(f'INPY=                       {INPYp}')
                print(f'INPA=                       {INPAp}')
                print(f'Runoff=                     {PBUFFIN[INPXp, INPYp]}')
                print(f'Final result PBUFFOUT({ISEQP, 1}) = {PBUFFOUT[ISEQP, 1]}  m3/s ')
                print('########################################################')

            return PBUFFOUT
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        #! Runoff interpolation & unit conversion (mm/dt -> m3/sec)
        if self.LINTERP:                    #! mass conservation using "input matrix table (inpmat)"
            self.D2RUNOFF               =    ROFF_INTERP  (CC_NMLIST, CM_NMLIST, PBUFF[:,:,1],log_filename, config['device'], Datatype)
            CC_VAR.D2RUNOFF[:,:]       =    self.D2RUNOFF[:, :]
            if CC_NMLIST.LROSPLIT:
                self.D2ROFSUB           =    ROFF_INTERP  (CC_NMLIST, CM_NMLIST, PBUFF[:,:,2],log_filename, config['device'], Datatype)
                CC_VAR.D2ROFSUB[:, :]  =    self.D2ROFSUB[:, :]
            else:
                self.D2ROFSUB           =   torch.zeros((CM_NMLIST.NSEQALL,1), dtype=Datatype.JPRB, device=config['device'])
                self.D2ROFSUB           =   Ftensor_2D(self.D2ROFSUB, start_row=1, start_col=1)
                CC_VAR.D2ROFSUB[:, :]  =   self.D2ROFSUB[:, :]
        else:
            print("The 'CONV_RESOL' code in 443-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
            print("The 'CONV_RESOL' code in 444-th Line for cmf_ctrl_forcing_mod.py is needed to improved")

        if CC_NMLIST.LWEVAP:
            print("The 'ROFF_INTERP' code in 447-th Line for cmf_ctrl_forcing_mod.py is needed to improved")
            print("The 'ROFF_INTERP' code in 448-th Line for cmf_ctrl_forcing_mod.py is needed to improved")

        return  CC_VAR
    def CMF_FORCING_END(self,log_filename):
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n!---------------------!\n")
            log_file.write(f"CMF::FORCING_END: Finalize forcing module\n")
            log_file.write(f"CMF::FORCING_END: end\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------


    # def CMFConfigCheck(self,log_filename):
    #     with open(log_filename, 'a') as log_file:
    #
    #         # Check for time step consistency
    #         if self.DT < 60 or int(self.DT) % 60 != 0:
    #             log_file.write(f"DT={self.DT} should be a multiple of 60. CaMa-Flood controls time by MINUTE.")
    #             raise ValueError("Stop: DT should be a multiple of 60.")
    #
    #         if int(self.DTIN) % int(self.DT) != 0:
    #             log_file.write(f"DTIN={self.DTIN}, DT={self.DT}. DTIN should be a multiple of DT.")
    #             raise ValueError("Stop: DTIN should be a multiple of DT.")
    #
    #         if self.LSEALEV and int(self.DTSL) % int(self.DT) != 0:
    #             log_file.write(f"DTSL={self.DTSL}, DT={self.DT}. DTSL should be a multiple of DT.")
    #             raise ValueError("Stop: DTSL should be a multiple of DT.")
    #
    #         # Check for physics options consistency
    #         if not self.LFPLAIN and not self.LKINE:
    #             log_file.write(
    #                 "LFPLAIN=False and LKINE=False. CAUTION: NO FLOODPLAIN option recommended to be used with kinematic wave (LKINE=True).")
    #
    #         if self.LKINE and self.LADPSTP:
    #             log_file.write(
    #                 "LKINE=True and LADPSTP=True. Adaptive time step recommended only with local inertial equation (LKINE=False).")
    #             log_file.write("Set appropriate fixed time step for Kinematic Wave.")
    #
    #         if self.LKINE and self.LPTHOUT:
    #             log_file.write(
    #                 "LKINE=True and LPATHOUT=True. Bifurcation channel flow only available with local inertial equation (LKINE=False).")
    #             raise ValueError("Stop: Bifurcation channel flow requires LKINE=False.")
    #
    #         if self.LGDWDLY and not self.LROSPLIT:
    #             log_file.write(
    #                 "LGDWDLY=True and LROSPLIT=False. Groundwater reservoir can only be active when runoff splitting is on.")
    #
    #         if self.LWEVAPFIX and not self.LWEVAP:
    #             log_file.write("LWEVAPFIX=True and LWEVAP=False. LWEVAPFIX can only be active if LWEVAP is active.")
    #             raise ValueError("Stop: LWEVAPFIX can only be active if LWEVAP is active.")
    #
    #         if self.LWEXTRACTRIV and not self.LWEVAP:
    #             log_file.write(
    #                 "LWEXTRACTRIV=True and LWEVAP=False. LWEXTRACTRIV can only be active if LWEVAP is active.")
    #             raise ValueError("Stop: LWEXTRACTRIV can only be active if LWEVAP is active.")
    #
    #         log_file.write("CMF::CONFIG_CHECK: end")
    #
    #     # --------------------------------------------------------------------------------------------------------------
