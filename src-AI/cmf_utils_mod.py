#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  Shared ulitity functions/subroutines for CaMa-Flood (python)
Licensed under the Apache License, Version 2.0.

! map related subroutines & functions
!-- vecP2mapR     : convert 1D vector data -> 2D map data (REAL*4)
!-- vecD2mapD    : convert 1D vector data -> 2D map data (REAL*8)
!-- mapR2vecD     : convert 2D map data -> 1D vector data (REAL*4)
!-- mapP2vecP    : convert 2D map data -> 1D vector data (REAL*8)
!-- mapI2vecI    : convert 2D map data -> 1D vector data (Integer)
!
! time related subroutines & functions
! -- MIN2DATE  : calculate DATE of KMIN from base time (YYYY0,MM0,DD0)
! -- DATE2MIN  : convert (YYYYMMDD,HHMM) to KMIN from base time (YYYY0,MM0,DD0)
! -- SPLITDATE : splite date (YYYYMMDD) to (YYYY,MM,DD)
! -- SPLITHOUR : split hour (HHMM) to (HH,MM)
! -- IMDAYS    : function to calculate days in a monty IMDAYS(IYEAR,IMON)
!
! endian conversion
!-- CONV_END    : Convert 2D Array endian (REAL4)
!-- CONV_ENDI   : Convert 2D Array endian (Integer)
!-- ENDIAN4R    : byte swap (REAL*4)
!-- ENDIAN4I    : byte swap (Integer)
!
! file I/O
!-- INQUIRE_FID : inruire unused file FID
!-- NCERROR     : netCDF I/O wrapper
!-- CMF_CheckNaN: check the value is NaN or not
"""
import  os
from fortran_tensor_3D import Ftensor_3D
from fortran_tensor_2D import Ftensor_2D
from fortran_tensor_1D import Ftensor_1D
import torch
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

def SPLITDATE(YYYYMMDD):
    # ============================
    YYYY  = YYYYMMDD // 10000  #
    MM = (YYYYMMDD % 10000) // 100  #
    DD = YYYYMMDD % 100  #
    # ============================
    return YYYY,MM,DD
def IMDAYS(IYEAR, IMON, LLEAPYR):
    """
    Returns the number of days in a given month and year.

    Returns:
    int: Number of days in the month.
    """
    ND = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Default days in the month
    IMDAYS = ND[IMON - 1]

    # Check for leap year adjustment in February
    if IMON == 2 and LLEAPYR:
        if (IYEAR % 400 == 0) or (IYEAR % 100 != 0 and IYEAR % 4 == 0):
            IMDAYS = 29
    return IMDAYS


class CMF_UTILS_MOD:
    def __init__(self,Datatype,  CC_NMLIST, CM_NMLISTT):
        # *** 2. default value
        self.JPIM       =       Datatype.JPIM
        self.JPRB       =       Datatype.JPRB
        self.JPRM       =       Datatype.JPRM
        self.JPRD       =       Datatype.JPRD
        self.DMIS       =       CC_NMLIST.DMIS
        self.RMIS       =       CC_NMLIST.RMIS
        self.NX         =       CC_NMLIST.NX
        self.NY         =       CC_NMLIST.NY
        self.NSEQMAX    =       getattr(CM_NMLISTT, 'NSEQMAX', 0)
        self.NSEQALL    =       getattr(CM_NMLISTT, 'NSEQALL', 0)
        self.LLEAPYR    =       CC_NMLIST.LLEAPYR
        self.D2MIN      =       1440


    def DATE2MIN(self, YYYYMMDD,HHMM,YYYY0,log_filename):
        #-----------------------------------------------------------------------------------------
        DATE2MIN = 0
        YYYY, MM, DD = SPLITDATE(YYYYMMDD)

        HH = HHMM // 100  # hour
        MI = HHMM - HH * 100  # minute

        # ============================
        with open(log_filename, 'a') as log_file:
            if YYYY < YYYY0:
                log_file.write(f"DATE2MIN: YYYY .LT. YYYY0: Date Problem: {YYYY,YYYY0}\n")
                raise ValueError(f"DATE2MIN: YYYY .LT. YYYY0: Date Problem {YYYY}, {YYYY0}")
            if MM < 1 or MM > 12:
                log_file.write(f"DATE2MIN: MM:    Date Problem: {YYYYMMDD, HHMM}\n")
                raise ValueError(f"DATE2MIN: MM:    Date Problem {YYYYMMDD}, {HHMM}")
            if DD < 1 or DD > IMDAYS(YYYY, MM, self.LLEAPYR):
                log_file.write(f"DATE2MIN: DD:    Date Problem: {YYYYMMDD, HHMM}\n")
                raise ValueError(f"DATE2MIN: DD:    Date Problem {YYYYMMDD}, {HHMM}")
            if HH < 0 or HH > 24:
                log_file.write(f"DATE2MIN: HH:    Date Problem: {YYYYMMDD, HHMM}\n")
                raise ValueError(f"DATE2MIN: HH:    Date Problem {YYYYMMDD}, {HHMM}")
            if MI < 0 or MI > 60:
                log_file.write(f"DATE2MIN: MI:    Date Problem: {YYYYMMDD, HHMM}\n")
                raise ValueError(f"DATE2MIN: MI:    Date Problem {YYYYMMDD}, {HHMM}")

            IY = YYYY0
            while IY < YYYY:
                for IM in range(1, 13):
                    DATE2MIN = DATE2MIN + IMDAYS(IY, IM, self.LLEAPYR) * self.D2MIN
                IY = IY + 1

            IM = 1
            while IM < MM:
                DATE2MIN = DATE2MIN + IMDAYS(IY, IM, self.LLEAPYR) * self.D2MIN
                IM = IM + 1

            DATE2MIN = DATE2MIN + (DD - 1) * self.D2MIN
            DATE2MIN = DATE2MIN + HH * 60 + MI

            return DATE2MIN

    # ==========================================================
    def SPLITDATE(self,YYYYMMDD):
        # sprit YYYYMMDD to (YYYY,MM,DD)

        # =================================================
        YYYY = YYYYMMDD // 10000  # 提取年份
        MM = (YYYYMMDD - YYYY * 10000) // 100  # 提取月份
        DD = YYYYMMDD - (YYYY * 10000 + MM * 100)  # 提取日期

        return YYYY, MM, DD

    # ==========================================================
    def SPLITHOUR(self,HHMM):
        # sprit HHMM to (HH, MI)

        # =================================================
        HH = HHMM // 100          # 提取小时
        MI = HHMM - HH * 100      # 提取分钟

        return HH, MI


    def mapR2vecD(self,R2TEMP,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        D2VAR       =       torch.zeros(NSEQMAX,1).clone().detach().to(dtype=self.JPRB, device=device)
        D2VAR       =       Ftensor_2D(D2VAR, start_row=1, start_col=1)
    # ------------------------------------------------------------------------------------------------------------------
        IX          =       I1SEQX.raw().long()
        IY          =       I1SEQY.raw().long()
        D2VAR[:,:]  =       R2TEMP[IX, IY].unsqueeze(1)
        return D2VAR
    def mapR2vecD_(self,R2TEMP,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        D2VAR       =       torch.zeros(NSEQMAX,1).clone().detach().to(dtype=self.JPRB, device=device)
        D2VAR       =       Ftensor_2D(D2VAR, start_row=1, start_col=1)
    # ------------------------------------------------------------------------------------------------------------------
        IX          =       I1SEQX.raw().long()
        IY          =       I1SEQY.raw().long()
        D2VAR[:,1]  =       R2TEMP[IX, IY]
        return D2VAR[:,1]

    def mapI2vecI(self,R2TEMP,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        D2VAR       =       torch.zeros(NSEQMAX, 1).clone().detach().to(dtype=self.JPIM, device=device)
        D2VAR       =       Ftensor_2D(D2VAR, start_row=1, start_col=1)
        # ------------------------------------------------------------------------------------------------------------------
        IX = I1SEQX.raw().long()
        IY = I1SEQY.raw().long()
        D2VAR[:, :] = R2TEMP[IX, IY].unsqueeze(1)
        return D2VAR

    def mapI2vecI_(self,R2TEMP,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        D2VAR       =       torch.zeros(NSEQMAX, 1).clone().detach().to(dtype=self.JPIM, device=device)
        D2VAR       =       Ftensor_2D(D2VAR, start_row=1, start_col=1)
        # ------------------------------------------------------------------------------------------------------------------
        IX = I1SEQX.raw().long()
        IY = I1SEQY.raw().long()
        D2VAR[:, 1] = R2TEMP[IX, IY]
        return D2VAR[:, 1]
    def MIN2DATE(self, IMIN, YYYY0, MM0, DD0):

        """!  Return YYYYMMDD and HHMM for IMIN"""

        NDAYS = IMIN // self.D2MIN  # days  in IMIN : 1440 = (minutes in a day)
        MI = IMIN % self.D2MIN
        HH = int(MI // 60)  # hours in IMIN
        MI = MI % 60  # mins  in IMIN

        YYYY = YYYY0.clone()
        MM = MM0.clone()
        DD = DD0.clone()
        NDM = IMDAYS(YYYY, MM, self.LLEAPYR)  # number of days in a month

        for ID in range(1, NDAYS + 1):
            DD += 1
            if DD > NDM:
                MM += 1
                DD = 1
                if MM > 12:
                    MM = 1
                    YYYY += 1
                NDM = IMDAYS(YYYY, MM, self.LLEAPYR)

        HHMM = HH * 100 + MI
        YYYYMMDD = YYYY * 10000 + MM * 100 + DD

        return YYYYMMDD, HHMM

    def vecD2mapR(self,D2VAR,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        R2TEMP = torch.ones((self.NX, self.NY),dtype=self.JPRM, device=device) * self.RMIS
        R2TEMP = Ftensor_2D(R2TEMP, start_row=1, start_col=1)
        # ------------------------------------------------------------------------------------------------------------------
        IX = I1SEQX.raw().long()
        IY = I1SEQY.raw().long()
        R2TEMP[IX, IY] = D2VAR[:NSEQMAX, 0].to(dtype=self.JPRM)
        return R2TEMP

    def vecP2mapP(self,P2VEC,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        P2MAP = torch.zeros((self.NX, self.NY), dtype=P2VEC.dtype, device=device)
        P2MAP = Ftensor_2D(P2MAP, start_row=1, start_col=1)

        # ------------------------------------------------------------------------------------------------------------------
        IX = I1SEQX.raw().long()
        IY = I1SEQY.raw().long()
        P2MAP[IX, IY] = P2VEC[:NSEQMAX, 0]
        return P2MAP

    def mapP2vecP(self,P2TEMP,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        P2VAR       =       torch.zeros(NSEQMAX,1).clone().detach().to(dtype=self.JPRD, device=device)
        P2VAR       =       Ftensor_2D(P2VAR, start_row=1, start_col=1)
    # ------------------------------------------------------------------------------------------------------------------
        IX          =       I1SEQX.raw().long()
        IY          =       I1SEQY.raw().long()
        P2VAR[:,:]  =       P2TEMP[IX, IY].unsqueeze(1)
        return P2VAR

    def mapP2vecD(self,P2TEMP,I1SEQX,I1SEQY,NSEQMAX,device):
        import torch

        D2VAR       =       torch.zeros(NSEQMAX,1).clone().detach().to(dtype=self.JPRB, device=device)
        D2VAR       =       Ftensor_2D(D2VAR, start_row=1, start_col=1)
    # ------------------------------------------------------------------------------------------------------------------
        IX          =       I1SEQX.raw().long()
        IY          =       I1SEQY.raw().long()
        D2VAR[:,:]  =       P2TEMP[IX, IY].unsqueeze(1)
        return D2VAR