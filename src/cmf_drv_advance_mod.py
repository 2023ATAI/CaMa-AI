#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@Co-author3: Cheng Zhang:  zc24@mails.jlu.edu.cn（Email）
@purpose:  Advance CaMa-Flood time integration   (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_DRV_ADVANCE : Advance integration for KSPETS
"""
import  os

import cmf_ctrl_physics_mod
from cmf_calc_diag_mod import CMF_DIAG_AVEMAX_OUTPUT, CMF_DIAG_GETAVE_OUTPUT, CMF_DIAG_RESET_OUTPUT

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'

def ADVANCE(KSTEPS,       CC_NMLIST ,     CM_NMLIST,    CT_NMLIST,     config,      Datatype,     CU,      CC_VAR,
            CO_NMLIST,    CR_NMLIST ,     device    ):
# ----------------------------------------------------------------------------------------------------------------------

#   *** get OMP thread number
#   $OMP PARALLEL
#   $ NTHREADS=OMP_GET_MAX_THREADS()
#   $OMP END PARALLEL

# ----------------------------------------------------------------------------------------------------------------------
    log_filename = config['RDIR'] + config['LOGOUT']

#   !*** START: time step loop
    for ISTEP in range(1, KSTEPS + 1):
        #   ------------------------------------------------------------------------------------------------------------
        # !*** 1. Set next time
        CT_NMLIST.CMF_TIME_NEXT(CC_NMLIST, log_filename, CU, config['device'])                   #  !! set KMINNEXT, JYYYYMMDD, JHHMM

        # !*** (optional)
        if CC_NMLIST.LSEALEV:
            raise NotImplementedError("LSEALEV is not supported in CaMa-PyTorch v1.0.")

        if CC_NMLIST.LTRACE:
            raise NotImplementedError("LTRACE is not supported in CaMa-PyTorch v1.0.")

        #   ------------------------------------------------------------------------------------------------------------
        # !*** 2. Advance model integration
        CC_VARS, CC_NMLIST, CM_NMLIST       =       (cmf_ctrl_physics_mod.CMF_PHYSICS_ADVANCE
                                                     (CC_NMLIST, CM_NMLIST, CT_NMLIST, log_filename, config['device'], Datatype, CU, CC_VAR, config))

        #   #ifdef sediment
        #       !*** 2b.  Advance sediment model integration
        #       IF( LSEDOUT .and. MOD(KSTEP,step_sed)==0 )THEN
        #           CALL cmf_calc_sedflw
        #       ENDIF
        #   #endif

        if CC_NMLIST.LTRACE:
            raise NotImplementedError("LTRACE is not supported in CaMa-PyTorch v1.0.")

        CC_VARS         =       CMF_DIAG_AVEMAX_OUTPUT  (CC_NMLIST,CC_VARS,CM_NMLIST,config['device'])   #   !! average & maximum calculation for output

        #   ------------------------------------------------------------------------------------------------------------
        #   !*** 3. Write output file (when needed)
        if CC_NMLIST.LOUTPUT and (CT_NMLIST.JHOUR % CC_NMLIST.IFRQ_OUT == 0) and (CT_NMLIST.JMIN == 0):
            #!*** average variable
            CC_VARS                         =          CMF_DIAG_GETAVE_OUTPUT(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, config['device'], Datatype)  #  !! average & maximum calculation for output: finalize

            #!*** write output data
            CO_NMLIST.CMF_OUTPUT_WRITE(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, Datatype, config['device'],CM_NMLIST,CU)

            if CC_NMLIST.LSEDOUT:
                raise NotImplementedError("LSEDOUT is not supported in CaMa-PyTorch v1.0.")

            #    ! --- Optional: text file output
            # CMF_OUTTXT_WRTE
            # CMF_DAMOUT_WRTE
            #
            # #   !*** reset variable
            CC_VARS                         =          CMF_DIAG_RESET_OUTPUT  (log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, CM_NMLIST, config['device'], Datatype)  #  !! average & maximum calculation for output: reset


        #   ------------------------------------------------------------------------------------------------------------
        #   !*** 4. Write restart file
        CR_NMLIST.CMF_RESTART_WRITE(log_filename, CT_NMLIST,CC_NMLIST, CM_NMLIST, CC_VARS, Datatype, CU, device, config)

        #   ------------------------------------------------------------------------------------------------------------
        #     !*** 5. Update current time      !! Update KMIN, IYYYYMMDD, IHHMM (to KMINNEXT, JYYYYMMDD, JHHMM)
        CT_NMLIST.CMF_TIME_UPDATE(log_filename)

        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::DRV_ADVANCE END: KSTEP, time (end of Tstep):  {CT_NMLIST.KSTEP}, {CT_NMLIST.JYYYYMMDD}, {CT_NMLIST.JHHMM}\n")
            log_file.flush()
            log_file.close()
    #   !*** END:time step loop
    return
