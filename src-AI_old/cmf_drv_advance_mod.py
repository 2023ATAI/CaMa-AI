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
import time

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
        # *** 0. get start CPU time
        ZTT0 = time.time()

        #   ------------------------------------------------------------------------------------------------------------
        # !*** 1. Set next time
        CT_NMLIST.CMF_TIME_NEXT(CC_NMLIST, log_filename, CU, config['device'])                   #  !! set KMINNEXT, JYYYYMMDD, JHHMM

        # !*** (optional)
        if CC_NMLIST.LSEALEV:
            print("The 'CMF_BOUNDARY_UPDATE' code in 47-th Line for cmf_drv_advance_mod.py is needed to improved")
            print("The 'CMF_BOUNDARY_UPDATE' code in 48-th Line for cmf_drv_advance_mod.py is needed to improved")

        if CC_NMLIST.LTRACE:
            print("The 'CMF_TRACER_DENSITY' code in 51-th Line for cmf_drv_advance_mod.py is needed to improved")
            print("The 'CMF_TRACER_DENSITY' code in 52-th Line for cmf_drv_advance_mod.py is needed to improved")

        #   ------------------------------------------------------------------------------------------------------------
        # !*** 2. Advance model integration
        CC_VARS, CC_NMLIST, CM_NMLIST       =       (cmf_ctrl_physics_mod.CMF_PHYSICS_ADVANCE
                                                     (CC_NMLIST, CM_NMLIST, CT_NMLIST, log_filename, config['device'], Datatype, CU, CC_VAR))

        #   #ifdef sediment
        #       !*** 2b.  Advance sediment model integration
        #       IF( LSEDOUT .and. MOD(KSTEP,step_sed)==0 )THEN
        #           CALL cmf_calc_sedflw
        #       ENDIF
        #   #endif

        if CC_NMLIST.LTRACE:
            print("The 'CMF_TRACER_FLUX' code in 67-th Line for cmf_drv_advance_mod.py is needed to improved")
            print("The 'CMF_TRACER_FLUX' code in 68-th Line for cmf_drv_advance_mod.py is needed to improved")

        # ! cost test
        if CC_NMLIST.MODTTEST:
            start_moduel = time.time()

        CC_VARS         =       CMF_DIAG_AVEMAX_OUTPUT  (CC_NMLIST,CC_VARS,CM_NMLIST,config['device'])   #   !! average & maximum calculation for output

        # ! cost test
        if CC_NMLIST.MODTTEST:
            end_moduel = time.time()
            CT_NMLIST.test_cost.setdefault('CMF_DIAG_AVEMAX_OUTPUT', []).append(end_moduel - start_moduel)

        ZTT1 = time.time()
        #   !$ ZTT1=OMP_GET_WTIME()

        #   ------------------------------------------------------------------------------------------------------------
        #   !*** 3. Write output file (when needed)
        if CC_NMLIST.LOUTPUT and (CT_NMLIST.JHOUR % CC_NMLIST.IFRQ_OUT == 0) and (CT_NMLIST.JMIN == 0):
            #!*** average variable
            CC_VARS                         =          CMF_DIAG_GETAVE_OUTPUT(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, config['device'], Datatype)  #  !! average & maximum calculation for output: finalize

            # ! cost test
            if CC_NMLIST.MODTTEST:
                start_moduel = time.time()

            #!*** write output data
            CO_NMLIST.CMF_OUTPUT_WRITE(log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, Datatype, config['device'],CM_NMLIST,CU)

            # ! cost test
            if CC_NMLIST.MODTTEST:
                end_moduel = time.time()
                CT_NMLIST.test_cost.setdefault('CMF_OUTPUT_WRITE', []).append(end_moduel - start_moduel)

            if CC_NMLIST.LSEDOUT:
                print("The 'cmf_sed_output' code in 78-th Line for cmf_drv_advance_mod.py is needed to improved")
                print("The 'cmf_sed_output' code in 79-th Line for cmf_drv_advance_mod.py is needed to improved")

            #    ! --- Optional: text file output
            # CMF_OUTTXT_WRTE
            # CMF_DAMOUT_WRTE
            #
            # #   !*** reset variable
            CC_VARS                         =          CMF_DIAG_RESET_OUTPUT  (log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, CM_NMLIST, config['device'], Datatype)  #  !! average & maximum calculation for output: reset


        #   ------------------------------------------------------------------------------------------------------------
        #   !*** 4. Write restart file
        # ! cost test
        if CC_NMLIST.MODTTEST:
            start_moduel = time.time()

        CR_NMLIST.CMF_RESTART_WRITE(log_filename, CT_NMLIST,CC_NMLIST, CM_NMLIST, CC_VARS, Datatype, CU, device, config)

        # ! cost test
        if CC_NMLIST.MODTTEST:
            end_moduel = time.time()
            CT_NMLIST.test_cost.setdefault('CMF_RESTART_WRITE', []).append(end_moduel - start_moduel)

        #   ------------------------------------------------------------------------------------------------------------
        #     !*** 5. Update current time      !! Update KMIN, IYYYYMMDD, IHHMM (to KMINNEXT, JYYYYMMDD, JHHMM)
        CT_NMLIST.CMF_TIME_UPDATE(log_filename)

        #   ------------------------------------------------------------------------------------------------------------
        #   !*** 6. Check CPU time
        ZTT2 = time.time()

        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::DRV_ADVANCE END: KSTEP, time (end of Tstep):  {CT_NMLIST.KSTEP}, {CT_NMLIST.JYYYYMMDD}, {CT_NMLIST.JHHMM}\n")
            log_file.write(f"Elapsed cpu time:  {ZTT2-ZTT0} Sec. // File output {ZTT2-ZTT1}\n")
            log_file.flush()
            log_file.close()
    #   !*** END:time step loop

    return ZTT0