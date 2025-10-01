#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  Manage prognostic/diagnostic variables in CaMa-Flood (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_PROG_INIT      : Initialize Prognostic variables (include restart data handling)
! -- CMF_DIAG_INIT      : Initialize Diagnostic variables
"""
import  os
import torch
import torch._dynamo
from fortran_tensor_2D import Ftensor_2D
from fortran_tensor_1D import Ftensor_1D

torch._dynamo.config.suppress_errors = True

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

class CMF_CTRL_VARS_MOD:
    """
    Created on  March  24  08:42 2025
    @author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
    @Co-author: Jian Hong; Gan Li
    @purpose:  Manage prognostic/diagnostic variables in CaMa-Flood (python)
    Licensed under the Apache License, Version 2.0.

    * CONTAINS:
    ! -- CMF_PROG_INIT      : Initialize Prognostic variables (include restart data handling)
    ! -- CMF_DIAG_INIT      : Initialize Diagnostic variables
    """
    def __init__(self,  JPRB,  JPRD):
        # --------------------------------------------------------------------------------------------------------------
        # ! Pointer was removed in v4.08 in order to keep simple codes when activating Single Precision Mode
        # !*** prognostics / state variables initial conditions
        # ------------------------------------------YOS_CMF_PROG-------------------------------------------------------
        # ! dammy variable for data handling
        self.D2DAMMY_TyPe                =           JPRB            # !! Dammy Array for unused variables
        self.D2COPY_TyPe                 =           JPRB            # !! Dammy Array for Float64/32 switch

        # --------------------------------------------------------------------------------------------------------------
        # !*** input runoff (interporlated)
        self.D2RUNOFF_TyPe               =           JPRB            # !! input runoff             [m3/s]
        self.D2ROFSUB_TyPe               =           JPRB            # !! input sub-surface runoff [m3/s]
        self.D2WEVAP_TyPe                =           JPRB            # !! input Evaporation [m3/s]

        # --------------------------------------------------------------------------------------------------------------
        # !*** river & floodpain
        # ! storage variables are always in double precision
        self.P2RIVSTO_TyPe               =           JPRD            # !! river      storage [m3]
        self.P2FLDSTO_TyPe               =           JPRD            # !! floodplain storage [m3]

        self.D2RIVOUT_TyPe               =           JPRB            # !! river      outflow [m3/s]
        self.D2FLDOUT_TyPe               =           JPRB            # !! floodplain outflow [m3/s]

        # --------------------------------------------------------------------------------------------------------------
        # !*** for implicit schemes of the local inertial equation
        self.D2RIVOUT_PRE_TyPe           =           JPRB            # !! river      outflow [m3/s] (prev t-step)
        self.D2RIVDPH_PRE_TyPe           =           JPRB            # !! river      depth   [m]    (prev t-step)
        self.D2FLDOUT_PRE_TyPe           =           JPRB            # !! floodplain outflow [m3/s] (prev t-step)
        self.D2FLDSTO_PRE_TyPe           =           JPRB            # !! floodplain storage [m3]   (prev t-step)

        # --------------------------------------------------------------------------------------------------------------
        # !! keep these variables even when LGDWDLY is not used. (for simplifying runoff calculation)
        self.P2GDWSTO_TyPe               =           JPRD            # !! ground water storage  [m3]
        self.D2GDWRTN_TyPe               =           JPRB            # !! Ground water return flow [m3/s]

        # --------------------------------------------------------------------------------------------------------------
        # !*** These have a different share, not part of the D2PROG array
        self.D1PTHFLW_TyPe               =           JPRB            # !! flood path outflow [m3/s]
        self.D1PTHFLW_PRE_TyPe           =           JPRB            # !! flood path outflow [m3/s] (prev t-step)

        # --------------------------------------------------------------------------------------------------------------
        # !*** dam variables
        self.P2DAMSTO_TyPe               =           JPRD            # !! reservoir storage [m3]
        self.P2DAMINF_TyPe               =           JPRD            # !! reservoir inflow [m3/s]; discharge before operation

        # --------------------------------------------------------------------------------------------------------------
        # !*** levee variables
        self.P2LEVSTO_TyPe               =           JPRD            # !! flood storage in protected side (storage betwen river & levee)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # ------------------------------------------YOS_CMF_DIAG--------------------------------------------------------
        # ! Pointer was removed in v4.12 in order to keep simple codes when activating Single Precision Mode
        # !*** Inst. diagnostics
        self.D2RIVINF_TyPe                 =           JPRB            # !! river      inflow   [m3/s] (from upstream)
        self.D2RIVDPH_TyPe                 =           JPRB            # !! river      depth    [m]
        self.D2RIVVEL_TyPe                 =           JPRB            # !! flow velocity       [m/s]

        self.D2FLDINF_TyPe                 =           JPRB            # !! floodplain inflow   [m3/s]
        self.D2FLDDPH_TyPe                 =           JPRB            # !! floodplain depth    [m]
        self.D2FLDFRC_TyPe                 =           JPRB            # !! flooded    fractipn [m2/m2]
        self.D2FLDARE_TyPe                 =           JPRB            # !! flooded    area     [m2]

        self.D1PTHFLWSUM_TyPe              =           JPRB            #  !! bifurcation flow (1D, not 2D variable), all layer sum
        self.D2PTHOUT_TyPe                 =           JPRB            # !! flood path outflow   [m3/s]
        self.D2PTHINF_TyPe                 =           JPRB            # !! flood path inflow   [m3/s]

        self.D2SFCELV_TyPe                 =           JPRB            # !! water surface elev  [m]    (elevtn - rivhgt + rivdph)
        self.D2OUTFLW_TyPe                 =           JPRB            # !! total outflow       [m3/s] (rivout + fldout)
        self.D2STORGE_TyPe                 =           JPRB            # !! total storage       [m3]   (rivsto + fldsto)

        self.D2OUTINS_TyPe                 =           JPRB            # !! instantaneous discharge [m3/s] (unrouted runoff)
        self.D2WEVAPEX_TyPe                =           JPRB            # !! Evaporation water extracted
        # --------------------------------------------------------------------------------------------------------------
        # !** local temporal variables in subroutine
        self.D2SFCELV_PRE_TyPe             =           JPRB
        self.D2DWNELV_PRE_TyPe             =           JPRB
        self.D2FLDDPH_PRE_TyPe             =           JPRB

        self.P2STOOUT_TyPe                 =           JPRB
        self.P2RIVINF_TyPe                 =           JPRB
        self.P2FLDINF_TyPe                 =           JPRB
        self.P2PTHOUT_TyPe                 =           JPRB
        self.D2RATE_TyPe                   =           JPRB
        # --------------------------------------------------------------------------------------------------------------
        # !*** Average diagnostics
        self.D2RIVOUT_AVG_TyPe             =           JPRB            # !! average river       discharge
        self.D2OUTFLW_AVG_TyPe             =           JPRB            # !! average total outflow       [m3/s] (rivout + fldout)  !! bugfix v362
        self.D2FLDOUT_AVG_TyPe             =           JPRB            # !! average floodplain  discharge
        self.D2RIVVEL_AVG_TyPe             =           JPRB            # !! average flow velocity
        self.D2PTHOUT_AVG_TyPe             =           JPRB            # !! flood pathway net outflow (2D)

        self.D2GDWRTN_AVG_TyPe             =           JPRB            # !! average ground water return flow
        self.D2RUNOFF_AVG_TyPe             =           JPRB            # !! average input runoff
        self.D2ROFSUB_AVG_TyPe             =           JPRB            # !! average input sub-surface runoff
        self.D2WEVAPEX_AVG_TyPe            =           JPRB           # !! average extracted water evaporation

        self.NADD_TyPe                     =           JPRB            # !! sum DT to calculate average
        # !*** Average diagnostics (1D)
        self.D1PTHFLW_AVG_TyPe             =           JPRB            # !! bifurcation channel flow (1D, not 2D variable)

        # --------------------------------------------------------------------------------------------------------------
        # !*** Daily max diagnostics
        self.D2OUTFLW_MAX_TyPe             =           JPRB            # !! max total outflow       [m3/s] (rivout + fldout)
        self.D2STORGE_MAX_TyPe             =           JPRB            # !! max total outflow       [m3/s] (rivout + fldout)
        self.D2RIVDPH_MAX_TyPe             =           JPRB            # !! max total outflow       [m3/s] (rivout + fldout)

        # --------------------------------------------------------------------------------------------------------------
        # !*** Global total
        # ! discharge calculation budget
        self.P0GLBSTOPRE_TyPe              =           JPRD            # !! global water storage      [m3] (befre flow calculation)
        self.P0GLBSTONXT_TyPe              =           JPRD            # !! global water storage      [m3] (after flow calculation)
        self.P0GLBSTONEW_TyPe              =           JPRD            # !! global water storage      [m3] (after runoff input)
        self.P0GLBRIVINF_TyPe              =           JPRD            # !! global inflow             [m3] (rivinf + fldinf)
        self.P0GLBRIVOUT_TyPe              =           JPRD            # !! global outflow            [m3] (rivout + fldout)

        # ! stage calculation budget
        self.P0GLBSTOPRE2_TyPe             =           JPRD            # !! global water storage      [m3] (befre stage calculation)
        self.P0GLBSTONEW2_TyPe             =           JPRD            # !! global water storage      [m3] (after stage calculation)
        self.P0GLBRIVSTO_TyPe              =           JPRD            # !! global river storage      [m3]
        self.P0GLBFLDSTO_TyPe              =           JPRD            # !! global floodplain storage [m3]
        self.P0GLBLEVSTO_TyPe              =           JPRD            # !! global protected-side storage [m3] (levee scheme)
        self.P0GLBFLDARE_TyPe              =           JPRD            # !! global flooded area       [m2]

        # --------------------------------------------------------------------------------------------------------------
        # !*** dam variable
        self.D2DAMINF_AVG_TyPe             =           JPRB            # !! average reservoir inflow [m3/s]  !!!added

        # --------------------------------------------------------------------------------------------------------------
        # !!!*** levee variables
        self.D2LEVDPH_TyPe                 =           JPRB            # !! flood depth in protected side (water depth betwen river & levee)

    def CMF_PROG_INIT(self,CM_NMLIST,CC_NMLIST,log_filename,device,Datatype):
        """
        CONTAINS
        !==========================================================
        !+ STORAGE_SEA_SURFACE: set initial storage, assuming water surface not lower than downstream sea surface elevation
        !+
        !+
        ! ==================================================
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def STORAGE_SEA_SURFACE(CM_NMLIST, Datatype, device):
            """
            set initial storage, assuming water surface not lower than downstream sea surface elevation
            """
            # --------------------------------------------------------------------------------------------------------------
            # ! set initial storage, assuming water surface not lower than downstream sea surface elevation
            self.DSEAELV            =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype.JPRB, device=device)
            self.DSEAELV            =           Ftensor_2D(self.DSEAELV, start_row=1, start_col=1)
            self.DDPH               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype.JPRB, device=device)
            self.DDPH               =           Ftensor_2D(self.DDPH, start_row=1, start_col=1)
            #  ! used to compute for CALC_ADPSTP code
            self.DDST               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=Datatype.JPRB, device=device)
            self.DDST               =           Ftensor_2D(self.DDST, start_row=1, start_col=1)
            # --------------------------------------------------------------------------------------------------------------
            #   For River Mouth Grid
            ID                      =           torch.arange(CM_NMLIST.NSEQRIV + 1, CM_NMLIST.NSEQALL + 1, dtype=torch.int32, device=device)   #   !! river mouth grid (NSEQALL, not NSEQRIV)
            self.DSEAELV[ID, :]     =           CM_NMLIST.D2DWNELV[ID, :]       #   !! downstream boundary elevation

            #   set initial water level to sea level if river bed is lower than sea level
            self.DDPH[ID, :]        =           torch.maximum(self.DSEAELV[ID, :] - CM_NMLIST.D2RIVELV[ID, :],
                                                              torch.zeros(self.DSEAELV[ID, :].shape,device=device))
            self.DDPH[ID, :]         =          torch.minimum(self.DDPH[ID, :], CM_NMLIST.D2RIVHGT[ID, :])
            self.P2RIVSTO[ID, :]     =          self.DDPH[ID, :] * CM_NMLIST.D2RIVLEN[ID, :] * CM_NMLIST.D2RIVWTH[ID, :]
            self.P2RIVSTO[ID, :]     =          torch.minimum(self.P2RIVSTO[ID, :],
                                                              CM_NMLIST.D2RIVSTOMAX[ID, :] * torch.tensor(1,dtype=Datatype.JPRD,device=device))
            self.D2RIVDPH_PRE[ID, :] =          self.DDPH[ID, :]

            # ---------------------------------------------------------------------------------------------------------------
            #  !! For Usual River Grid (from downstream to upstream). OMP cannot be applied

            for IESQ_ in reversed(range(CM_NMLIST.NSEQRIV)):
                IESQ                             =          IESQ_ + 1
                JSEQ                             =          CM_NMLIST.I1NEXT[IESQ]
                DSEAELV                          =          CM_NMLIST.D2RIVELV[JSEQ, 1] + self.D2RIVDPH_PRE[JSEQ, 1]

                #   set initial water level to sea level if river bed is lower than sea level
                self.DDPH[IESQ, 1]               =          torch.maximum(DSEAELV - CM_NMLIST.D2RIVELV[IESQ, 1], torch.tensor(0,dtype=Datatype.JPRB,device=device))
                self.DDPH[IESQ, 1]               =          torch.minimum(self.DDPH[IESQ, 1], CM_NMLIST.D2RIVHGT[IESQ, 1])

                self.P2RIVSTO[IESQ, 1]           =          (self.DDPH[IESQ, 1] * CM_NMLIST.D2RIVLEN[IESQ, 1] * CM_NMLIST.D2RIVWTH[IESQ, 1])
                self.P2RIVSTO[IESQ, 1]           =          torch.minimum(self.P2RIVSTO[IESQ, 1],
                                                                    CM_NMLIST.D2RIVSTOMAX[IESQ, 1] * torch.tensor(1,dtype=Datatype.JPRD,device=device))
                self.D2RIVDPH_PRE[IESQ, 1]       =          self.DDPH[IESQ, 1]
            return
        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------D2RUNOFF-------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n******************************!\n")

            log_file.write(f"CMF::PROG_INIT: prognostic variable initialization\n")
            log_file.flush()
            log_file.close()

        #   *** 1. ALLOCATE
        #   runoff input
        self.D2RUNOFF                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RUNOFF_TyPe,device=device)
        self.D2RUNOFF                   =           Ftensor_2D(self.D2RUNOFF, start_row=1, start_col=1)
        self.D2RUNOFF_year              =           torch.zeros((365,CM_NMLIST.NSEQMAX), dtype=self.D2RUNOFF_TyPe,device=device)
        self.D2RUNOFF_year              =           Ftensor_2D(self.D2RUNOFF_year, start_row=1, start_col=1)
        self.D2ROFSUB                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_TyPe,device=device)
        self.D2ROFSUB                   =           Ftensor_2D(self.D2ROFSUB, start_row=1, start_col=1)
        self.D2ROFSUB_year              =           torch.zeros((365,CM_NMLIST.NSEQMAX), dtype=self.D2ROFSUB_TyPe,device=device)
        self.D2ROFSUB_year              =           Ftensor_2D(self.D2ROFSUB_year, start_row=1, start_col=1)

        #   ! river+floodplain storage
        self.P2RIVSTO                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.P2RIVSTO_TyPe,device=device)
        self.P2RIVSTO                   =           Ftensor_2D(self.P2RIVSTO, start_row=1, start_col=1)
        self.P2FLDSTO                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.P2FLDSTO_TyPe,device=device)
        self.P2FLDSTO                   =           Ftensor_2D(self.P2FLDSTO, start_row=1, start_col=1)

        #   ! discharge calculation
        self.D2RIVOUT                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVOUT_TyPe,device=device)
        self.D2RIVOUT                   =           Ftensor_2D(self.D2RIVOUT, start_row=1, start_col=1)
        self.D2FLDOUT                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDOUT_TyPe,device=device)
        self.D2FLDOUT                   =           Ftensor_2D(self.D2FLDOUT, start_row=1, start_col=1)
        self.D2RIVOUT_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVOUT_PRE_TyPe,device=device)
        self.D2RIVOUT_PRE               =           Ftensor_2D(self.D2RIVOUT_PRE, start_row=1, start_col=1)
        self.D2FLDOUT_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDOUT_PRE_TyPe,device=device)
        self.D2FLDOUT_PRE               =           Ftensor_2D(self.D2FLDOUT_PRE, start_row=1, start_col=1)
        self.D2RIVDPH_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVDPH_PRE_TyPe,device=device)
        self.D2RIVDPH_PRE               =           Ftensor_2D(self.D2RIVDPH_PRE, start_row=1, start_col=1)
        self.D2FLDSTO_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDSTO_PRE_TyPe,device=device)
        self.D2FLDSTO_PRE               =           Ftensor_2D(self.D2FLDSTO_PRE, start_row=1, start_col=1)

        if CC_NMLIST.LPTHOUT:                   #   !! additional prognostics for bifurcation scheme
            self.D1PTHFLW_aAVG               =           torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV), dtype=self.D1PTHFLW_TyPe, device=device)
            self.D1PTHFLW_aAVG               =           Ftensor_2D(self.D1PTHFLW_aAVG, start_row=1, start_col=1)
            self.D1PTHFLWSUM_aAVG           =           torch.zeros((CM_NMLIST.NPTHOUT), dtype=self.D1PTHFLW_PRE_TyPe, device=device)
            self.D1PTHFLWSUM_aAVG           =           Ftensor_1D(self.D1PTHFLWSUM_aAVG, start_index=1)
        if CC_NMLIST.LDAMOUT:                   #   !! additional prognostics for reservoir operation
            self.P2DAMSTO               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.P2DAMSTO_TyPe, device=device)
            self.P2DAMSTO               =           Ftensor_2D(self.P2DAMSTO, start_row=1, start_col=1)
            self.P2DAMINF               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.P2DAMINF_TyPe, device=device)
            self.P2DAMINF               =           Ftensor_2D(self.P2DAMINF, start_row=1, start_col=1)
        if CC_NMLIST.LLEVEE:                   #   !! additional prognostics for LLEVEE
            self.P2LEVSTO               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.P2LEVSTO_TyPe, device=device)
            self.P2LEVSTO               =           Ftensor_2D(self.P2LEVSTO, start_row=1, start_col=1)

        # Used in ECMWF
        if CC_NMLIST.LWEVAP:
            self.D2WEVAP                =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2WEVAP_TyPe, device=device)
            self.D2WEVAP                =           Ftensor_2D(self.D2WEVAP, start_row=1, start_col=1)

        # keep these variables even when LGDWDLY is not used.
        self.P2GDWSTO                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.P2GDWSTO_TyPe, device=device)
        self.P2GDWSTO                   =           Ftensor_2D(self.P2GDWSTO, start_row=1, start_col=1)
        self.D2GDWRTN                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2GDWRTN_TyPe, device=device)
        self.D2GDWRTN                   =           Ftensor_2D(self.D2GDWRTN, start_row=1, start_col=1)

        # !*** These have a different share, not part of the D2PROG array
        self.D1PTHFLW                   =           torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV), dtype=self.D1PTHFLW_TyPe,device=device)     #   !! flood path outflow [m3/s]
        self.D1PTHFLW                   =           Ftensor_2D(self.D1PTHFLW, start_row=1, start_col=1)
        self.D1PTHFLW_PRE               =           torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV), dtype=self.D1PTHFLW_PRE_TyPe,device=device)     #   !! flood path outflow [m3/s] (prev t-step)
        self.D1PTHFLW_PRE               =           Ftensor_2D(self.D1PTHFLW_PRE, start_row=1, start_col=1)


        # dammy variable for data handling
        # !! Float64/32 switch (Dammy for unused var)
        self.D2DAMMY                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2DAMMY_TyPe, device=device)
        self.D2DAMMY                   =           Ftensor_2D(self.D2DAMMY, start_row=1, start_col=1)
        # !! Float64/32 switch (Dammy for output)
        self.D2COPY                    =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2COPY_TyPe, device=device)
        self.D2COPY                    =           Ftensor_2D(self.D2COPY, start_row=1, start_col=1)

        # --------------------------------------------------------------------------------------------------------------
        # 2. set initial water surface elevation to sea surface level
        with open(log_filename, 'a') as log_file:
            log_file.write(f"PROG_INIT: fill channels below downstream boundary\n")
            log_file.flush()
            log_file.close()
        STORAGE_SEA_SURFACE     (CM_NMLIST, Datatype, device)


        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::PROG_INIT: end\n")
            log_file.flush()
            log_file.close()

        return
    def CMF_DIAG_INIT(self,CM_NMLIST,CC_NMLIST,log_filename,device):
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n******************************!\n")

            log_file.write(f"CMF::DIAG_INIT: initialize diagnostic variables\n")
            log_file.flush()
            log_file.close()

        # 1. snapshot 2D diagnostics
        self.D2RIVINF                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVINF_TyPe,device=device)
        self.D2RIVINF                   =           Ftensor_2D(self.D2RIVINF, start_row=1, start_col=1)
        self.D2RIVDPH                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVDPH_TyPe,device=device)
        self.D2RIVDPH                   =           Ftensor_2D(self.D2RIVDPH, start_row=1, start_col=1)
        self.D2RIVVEL                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVVEL_TyPe,device=device)
        self.D2RIVVEL                   =           Ftensor_2D(self.D2RIVVEL, start_row=1, start_col=1)
        self.D2FLDINF                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDINF_TyPe,device=device)
        self.D2FLDINF                   =           Ftensor_2D(self.D2FLDINF, start_row=1, start_col=1)
        self.D2FLDDPH                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDDPH_TyPe,device=device)
        self.D2FLDDPH                   =           Ftensor_2D(self.D2FLDDPH, start_row=1, start_col=1)
        self.D2FLDFRC                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDFRC_TyPe,device=device)
        self.D2FLDFRC                   =           Ftensor_2D(self.D2FLDFRC, start_row=1, start_col=1)
        self.D2FLDARE                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDARE_TyPe,device=device)
        self.D2FLDARE                   =           Ftensor_2D(self.D2FLDARE, start_row=1, start_col=1)
        self.D2PTHOUT                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2PTHOUT_TyPe,device=device)
        self.D2PTHOUT                   =           Ftensor_2D(self.D2PTHOUT, start_row=1, start_col=1)
        self.D2PTHINF                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2PTHINF_TyPe,device=device)
        self.D2PTHINF                   =           Ftensor_2D(self.D2PTHINF, start_row=1, start_col=1)
        self.D2SFCELV                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2SFCELV_TyPe,device=device)
        self.D2SFCELV                   =           Ftensor_2D(self.D2SFCELV, start_row=1, start_col=1)
        self.D2OUTFLW                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2OUTFLW_TyPe,device=device)
        self.D2OUTFLW                   =           Ftensor_2D(self.D2OUTFLW, start_row=1, start_col=1)
        self.D2STORGE                   =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2STORGE_TyPe,device=device)
        self.D2STORGE                   =           Ftensor_2D(self.D2STORGE, start_row=1, start_col=1)

        self.D1PTHFLWSUM                =           torch.zeros((CM_NMLIST.NSEQMAX), dtype=self.D2STORGE_TyPe,device=device)
        self.D1PTHFLWSUM                =           Ftensor_1D(self.D1PTHFLWSUM, start_index=1)
        self.D1PTHFLW                   =           torch.zeros((CM_NMLIST.NPTHOUT,CM_NMLIST.NPTHLEV), dtype=self.D2STORGE_TyPe,device=device)
        self.D1PTHFLW                   =           Ftensor_2D(self.D1PTHFLW, start_row=1, start_col=1)

        if  CC_NMLIST.LLEVEE:
            self.D2LEVDPH               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2LEVDPH_TyPe,device=device)
            self.D2LEVDPH               =           Ftensor_2D(self.D2LEVDPH, start_row=1, start_col=1)
        if  CC_NMLIST.LWEVAP:
            self.D2WEVAPEX              =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2WEVAPEX_TyPe,device=device)
            self.D2WEVAPEX              =           Ftensor_2D(self.D2WEVAPEX, start_row=1, start_col=1)
        if CC_NMLIST.LOUTINS:
            self.D2OUTINS               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2OUTINS_TyPe,device=device)
            self.D2OUTINS               =           Ftensor_2D(self.D2OUTINS, start_row=1, start_col=1)
        # --------------------------------------------------------------------------------------------------------------
        # !** local temporal variables in subroutine

        # --------------------------------------------------------------------------------------------------------------
        # 2a. time-average 2D diagnostics
        self.D2RIVOUT_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVOUT_AVG_TyPe,device=device)
        self.D2FLDOUT_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDOUT_AVG_TyPe,device=device)
        self.D2OUTFLW_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2OUTFLW_AVG_TyPe,device=device)
        self.D2RIVVEL_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVVEL_AVG_TyPe,device=device)
        self.D2PTHOUT_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2PTHOUT_AVG_TyPe,device=device)
        self.D2GDWRTN_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2GDWRTN_AVG_TyPe,device=device)
        self.D2RUNOFF_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RUNOFF_AVG_TyPe,device=device)
        self.D2ROFSUB_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)

        self.D2STORGE_aMAX           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)
        self.D2OUTFLW_aMAX           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)
        self.D2RIVDPH_aMAX           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)


        if CC_NMLIST.LDAMOUT:
            self.D2DAMINF_aAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2DAMINF_AVG_TyPe,device=device)
        if CC_NMLIST.LWEVAP:
            self.D2WEVAPEX_aAVG          =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2WEVAPEX_AVG_TyPe,device=device)

        self.NADD_adp                   =           0
        self.NADD_out                   =           0

        # 2b time-average 1D Diagnostics (bifurcation channel)
        self.D1PTHFLW_aAVG               =           torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV), dtype=self.D1PTHFLW_AVG_TyPe, device=device)
        # --------------------------------------------------------------------------------------------------------------
        # !============
        # !*** 3a. time-average 2D diagnostics for output
        self.D2RIVOUT_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVOUT_AVG_TyPe,device=device)
        self.D2FLDOUT_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDOUT_AVG_TyPe,device=device)
        self.D2OUTFLW_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2OUTFLW_AVG_TyPe,device=device)
        self.D2RIVVEL_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RIVVEL_AVG_TyPe,device=device)
        self.D2PTHOUT_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2PTHOUT_AVG_TyPe,device=device)
        self.D2GDWRTN_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2GDWRTN_AVG_TyPe,device=device)
        self.D2RUNOFF_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2RUNOFF_AVG_TyPe,device=device)
        self.D2ROFSUB_oAVG           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)

        # !*** ab time-average 1D Diagnostics (bifurcation channel)
        self.D1PTHFLW_oAVG               =           torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV), dtype=self.D1PTHFLW_AVG_TyPe, device=device)
        # !*** 3c. Maximum 2D Diagnostics
        self.D2STORGE_oMAX           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)
        self.D2OUTFLW_oMAX           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)
        self.D2RIVDPH_oMAX           =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2ROFSUB_AVG_TyPe,device=device)


        # --------------------------------------------------------------------------------------------------------------
        # !*** 4. Global total
        # ! discharge calculation budget
        self.P0GLBSTOPRE                =           torch.tensor(0, dtype=self.P0GLBSTOPRE_TyPe, device=device)
        self.P0GLBSTONXT                =           torch.tensor(0, dtype=self.P0GLBSTONXT_TyPe, device=device)
        self.P0GLBSTONEW                =           torch.tensor(0, dtype=self.P0GLBSTONEW_TyPe, device=device)
        self.P0GLBRIVINF                =           torch.tensor(0, dtype=self.P0GLBRIVINF_TyPe, device=device)
        self.P0GLBRIVOUT                =           torch.tensor(0, dtype=self.P0GLBRIVOUT_TyPe, device=device)

        # ! stage calculation budget
        self.P0GLBSTOPRE2               =           torch.tensor(0, dtype=self.P0GLBSTOPRE2_TyPe, device=device)
        self.P0GLBSTONEW2               =           torch.tensor(0, dtype=self.P0GLBSTONEW2_TyPe, device=device)
        self.P0GLBRIVSTO                =           torch.tensor(0, dtype=self.P0GLBRIVSTO_TyPe, device=device)
        self.P0GLBFLDSTO                =           torch.tensor(0, dtype=self.P0GLBFLDSTO_TyPe, device=device)
        self.P0GLBLEVSTO                =           torch.tensor(0, dtype=self.P0GLBLEVSTO_TyPe, device=device)
        self.P0GLBFLDARE                =           torch.tensor(0, dtype=self.P0GLBFLDARE_TyPe, device=device)

        # --------------------------------------------------------------------------------------------------------------
        # !*** dam variable
        self.P0GLBFLDARE                =           torch.tensor(0, dtype=self.P0GLBFLDARE_TyPe, device=device)

        # --------------------------------------------------------------------------------------------------------------
        # !! temporally variables in subroutines
        self.D2SFCELV_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2SFCELV_PRE_TyPe, device=device)
        self.D2SFCELV_PRE               =           Ftensor_2D(self.D2SFCELV_PRE, start_row=1, start_col=1)
        self.D2DWNELV_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2DWNELV_PRE_TyPe, device=device)
        self.D2DWNELV_PRE               =           Ftensor_2D(self.D2DWNELV_PRE, start_row=1, start_col=1)
        self.D2FLDDPH_PRE               =           torch.zeros((CM_NMLIST.NSEQMAX, 1), dtype=self.D2FLDDPH_TyPe, device=device)
        self.D2FLDDPH_PRE               =           Ftensor_2D(self.D2FLDDPH_PRE, start_row=1, start_col=1)



        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::DIAG_INIT: end\n")
            log_file.flush()
            log_file.close()
        return