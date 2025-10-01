#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@Co-author3: Cheng Zhang:  zc24@mails.jlu.edu.cn（Email）
@purpose:  Initialize/Finalize CaMa-Flood Model (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_DRV_INPUT    : Set namelist & logfile
! -- CMF_DRV_INIT     : Initialize        CaMa-Flood
! -- CMF_DRV_END      : Finalize          CaMa-Flood
"""
def CMF_DRV_INPUT(config,    Datatype):
    """
    ! Read setting from namelist ("input_flood.nam" as default)
    ! -- Called from CMF_DRV_INIT
    """

    from cmf_ctrl_nmlist_mod import CMF_CTRL_NMLIST_MOD
    from cmf_ctrl_time_mod import CMF_CTRL_TIME_MOD
    from cmf_ctrl_restart_mod import CMF_RESTART_NMLIST_MOD
    from cmf_ctrl_output_mod import CMF_OUTPUT_NMLIST_MOD
    from cmf_ctrl_forcing_mod import CMF_FORCING_NMLIST_MOD


    # DT                          =       3600
    # ------------------------------------------------------------------------------------------------------------------
    #!*** 1. CaMa-Flood configulation namelist
    CC_NMLIST                    =       CMF_CTRL_NMLIST_MOD             (config,    Datatype)
    CC_NMLIST.log_settings                                               (config)

    CT_NMLIST                    =       CMF_CTRL_TIME_MOD               (config,    Datatype)
    CT_NMLIST.CMF_TIME_NMLIST                                           (config,    CC_NMLIST)

    # --------------------------------------------------------------------------------------------------------------
    #!*** 2. read namelist for each module
    CF_NMLIST                    =       CMF_FORCING_NMLIST_MOD          (config,    Datatype,  CC_NMLIST)
    CT_NMLIST                    =       CF_NMLIST.CMF_FORCING_NMLIST    (config,  CT_NMLIST,  CC_NMLIST)

    if CC_NMLIST.LSEALEV:
        print("The 'CMF_BOUNDARY_NMLIST' code in 50-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_BOUNDARY_NMLIST' code in 51-th Line for cmf_drv_control_mod.py is needed to improved")

    CR_NMLIST                     =       CMF_RESTART_NMLIST_MOD          ()
    CR_NMLIST.CMF_RESTART_NMLIST                                          (config, CC_NMLIST)

    if CC_NMLIST.LDAMOUT:
        print("The 'CMF_DAMOUT_NMLIST' code in 57th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_DAMOUT_NMLIST' code in 58-th Line for cmf_drv_control_mod.py is needed to improved")

    if CC_NMLIST.LSEALEV:
        print("The 'CMF_LEVEE_NMLIST' code in 61-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_LEVEE_NMLIST' code in 62-th Line for cmf_drv_control_mod.py is needed to improved")

    if CC_NMLIST.LOUTPUT:
        CO_NMLIST               =       CMF_OUTPUT_NMLIST_MOD           (config, Datatype)
        CO_NMLIST.CMF_OUTPUT_NMLIST                                     (config, CC_NMLIST)

    if CC_NMLIST.LSEDOUT:
        print("The 'cmf_sed_nmlist' code in 69-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'cmf_sed_nmlist' code in 70-th Line for cmf_drv_control_mod.py is needed to improved")

    log_filename = config['RDIR'] + config['LOGOUT']
    with open(log_filename, 'a') as log_file:
        log_file.write("CMF::DRV_INPUT: end reading namelist\n")
        log_file.flush()
        log_file.close()

    # !*** 3. check configulation conflicts
    CC_NMLIST.CMF_CONFIG_CHECK(log_filename)

    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DRV_INPUT: finished\n")
        log_file.write(f"******************************!\n")
        log_file.write(f"\n")
        log_file.flush()
        log_file.close()

    return (CC_NMLIST,      CT_NMLIST,         CF_NMLIST,   CR_NMLIST,       CO_NMLIST)
    # ------------------------------------------------------------------------------------------------------------------
    #!*** 1b. INITIALIZATION
def CMF_DRV_INIT(CC_NMLIST,             CT_NMLIST,              CM_NMLIST,      CF_NMLIST,          CR_NMLIST,
                 CO_NMLIST,             CU,                     config,                 Datatype,                       ):
    """
    ! Initialize CaMa-Flood
    ! -- Called from CMF_DRV_INIT
    """
    from cmf_ctrl_vars_mod import CMF_CTRL_VARS_MOD
    import cmf_ctrl_physics_mod
    import time
    log_filename = config['RDIR'] + config['LOGOUT']
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n******************************!\n")
        log_file.write(f"CMF::DRV_INIT: initialization start\n")
        log_file.flush()
        log_file.close()

    # 0. get start CPU time
    start_time                          =          time.time()

    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DRV_INIT: (1) Set Time\n")
        log_file.flush()
        log_file.close()

    # 1a. Set time related
    CC_NMLIST                           =          CT_NMLIST.CMF_TIME_INIT(CC_NMLIST,        log_filename,        CU)

    # 2c. Optional levee scheme initialization
    if CC_NMLIST.LLEVEE:
        print("The 'CMF_LEVEE_INIT' code in 154-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_LEVEE_INIT' code in 155-th Line for cmf_drv_control_mod.py is needed to improved")

    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        # Write settings to log
        log_file.write(f"CMF::DRV_INIT: (3) Set output & forcing modules\n")
        log_file.write("\n!---------------------!\n")

    # 3a. Create Output files
    if CC_NMLIST.LOUTPUT:
        CO_NMLIST.CMF_OUTPUT_INIT   (CC_NMLIST,          log_filename,            CM_NMLIST,           CT_NMLIST,
                                     config)

    # 3b. Initialize forcing data
    CC_NMLIST       =       (CF_NMLIST.CMF_FORCING_INIT
                             (CC_NMLIST,   CT_NMLIST,   CU,     log_filename,    CM_NMLIST,     Datatype))

    # 3b. Initialize dynamic sea level boundary data
    if CC_NMLIST.LSEALEV:
        print("The 'CMF_BOUNDARY_INIT' code in 174-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_BOUNDARY_INIT' code in 175-th Line for cmf_drv_control_mod.py is needed to improved")
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DRV_INIT: (4) Allocate prog & diag vars & initialize\n")
        log_file.flush()
        log_file.close()

    # 4a. Set initial prognostic variables
    CC_VAR       =          (CMF_CTRL_VARS_MOD
                              (Datatype.JPRB,       Datatype.JPRD                                          ))
    CC_VAR.CMF_PROG_INIT      (CM_NMLIST,   CC_NMLIST,  log_filename,   config['device'],   Datatype)

    # 4b. Initialize (allocate) diagnostic arrays
    CC_VAR.CMF_DIAG_INIT      (CM_NMLIST,   CC_NMLIST,  log_filename,   config['device'])

    # !v4.03 CALC_FLDSTG for zero storage restart
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Initialize start CMF_PHYSICS_FLDSTG\n")
        log_file.flush()
        log_file.close()
    # ----------------------------------------------------------------------------------------------------------------------
    #     for speed up

    CC_VAR.P2RIVSTO = CC_VAR.P2RIVSTO[:,:]
    CC_VAR.P2FLDSTO = CC_VAR.P2FLDSTO.raw()
    CM_NMLIST.D2RIVSTOMAX = CM_NMLIST.D2RIVSTOMAX.raw()
    CM_NMLIST.D2RIVWTH = CM_NMLIST.D2RIVWTH.raw()
    CM_NMLIST.D2GRAREA = CM_NMLIST.D2GRAREA.raw()
    CM_NMLIST.D2RIVLEN = CM_NMLIST.D2RIVLEN.raw()
    CM_NMLIST.D2FLDSTOMAX = CM_NMLIST.D2FLDSTOMAX.raw()
    CM_NMLIST.D2FLDGRD = CM_NMLIST.D2FLDGRD.raw()
    CC_VAR.D2FLDDPH = CC_VAR.D2FLDDPH.raw()
    CC_VAR.D2FLDFRC = CC_VAR.D2FLDFRC.raw()
    CC_VAR.D2FLDARE = CC_VAR.D2FLDARE.raw()
    CC_VAR.D2RIVDPH = CC_VAR.D2RIVDPH.raw()

    # ----------------------------------------------------------------------------------------------------------------------
    CM_NMLIST,  CC_VAR  =      (cmf_ctrl_physics_mod.CMF_PHYSICS_FLDSTG
                                                        (CM_NMLIST, CC_NMLIST,  CC_VAR, config['device'],   Datatype))
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Initialize end CMF_PHYSICS_FLDSTG\n")
        log_file.flush()
        log_file.close()
    #   *** 4c. Restart file
    if CC_NMLIST.LRESTART:
        CR_NMLIST.CMF_RESTART_INIT(config, CC_NMLIST, CC_VAR, config['device'], CM_NMLIST, Datatype, CU)

    # *** 4d. Optional reservoir initialization
    if CC_NMLIST.LDAMOUT:
        print("The 'CMF_DAMOUT_INIT' code in 209-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_DAMOUT_INIT' code in 210-th Line for cmf_drv_control_mod.py is needed to improved")

    #  *** 4e. Optional sediment initialization
    if CC_NMLIST.LSEDOUT:
        print("The 'cmf_sed_init' code in 214-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'cmf_sed_init' code in 215-th Line for cmf_drv_control_mod.py is needed to improved")

    # ------------------------------------------------------------------------------------------------------------------
     #!** v4.03 CALC_FLDSTG moved to the top of CTRL_PHYSICS for strict restart configulation (Hatono & Yamazaki)

    #!*** 5 reconstruct previous t-step flow (if needed)
    if CC_NMLIST.LRESTART and CC_NMLIST.LSTOONLY:
        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::DRV_INIT: (5a) set flood stage at initial condition\n")
            log_file.flush()
            log_file.close()
        CM_NMLIST,  CC_VAR = cmf_ctrl_physics_mod.CMF_PHYSICS_FLDSTG (CM_NMLIST, CC_NMLIST, CC_VAR, log_filename)
        print("The 'CMF_CALC_OUTPRE' code in 228-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_CALC_OUTPRE' code in 229-th Line for cmf_drv_control_mod.py is needed to improved")

    # !*** 5b save initial storage if LOUTINI specified
    if CC_NMLIST.LOUTINI and CC_NMLIST.LOUTPUT:
        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::DRV_INIT: (5b) write initial condition\n")
            log_file.flush()
            log_file.close()
        print("The 'CMF_OUTPUT_WRITE' code in 237-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_OUTPUT_WRITE' code in 238-th Line for cmf_drv_control_mod.py is needed to improved")

    # ------------------------------------------------------------------------------------------------------------------.

    #!*** get initialization end time time
    end_time = time.time()
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DRV_INIT: initialization finished:\n")
        log_file.write(f"Elapsed cpu time (Init){end_time - start_time}Seconds\n")
        log_file.write(f"CMF::DRV_INIT: end\n")
        log_file.write(f"***********************************\n")
        log_file.flush()
        log_file.close()

    return      (CC_NMLIST,             CT_NMLIST,             CM_NMLIST,       CF_NMLIST,          CR_NMLIST,
                 CO_NMLIST,             CC_VAR)

def CMF_DRV_END(config,CC_NMLIST,CF_NMLIST,CM_NMLIST,CO_NMLIST,CT_NMLIST,ZTT0):
    import time

    """
    ! Initialize CaMa-Flood
    ! -- Called from CMF_DRV_INIT
    """
    log_filename = config['RDIR'] + config['LOGOUT']

    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n!******************************!\n")
        log_file.write(f"CMF::DRV_END: finalize forcing & output modules\n")
        log_file.flush()
        log_file.close()
    CF_NMLIST.CMF_FORCING_END           (log_filename)
    if CC_NMLIST.LOUTPUT:
        CO_NMLIST.CMF_OUTPUT_END        (log_filename, CM_NMLIST)
        #ifdef sediment
        #IF( LSEDOUT ) call sediment_output_end
        ##endif
    if CC_NMLIST.LSEALEV:
        print("The 'CMF_BOUNDARY_END' code in 274-th Line for cmf_drv_control_mod.py is needed to improved")
        print("The 'CMF_BOUNDARY_END' code in 275-th Line for cmf_drv_control_mod.py is needed to improved")
    ZTT2 = time.time()
    elapsed = ZTT2 - ZTT0
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n!******************************!\n")
        log_file.write(f"CMF::DRV_END: simulation finished in: {elapsed:.2f} Seconds\n")
        log_file.write(f"CMF::DRV_END: close logfile\n")
        log_file.write(f"CMF::===== CALCULATION END =====\n")
        log_file.flush()
        log_file.close()
    # ! cost test
    if CC_NMLIST.MODTTEST:
        for module_name, time_list in CT_NMLIST.test_cost.items():
            if isinstance(time_list, list) and time_list:
                avg_time = sum(time_list) / len(time_list)
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"\n!******************************!\n")
                    log_file.write(f"CMF:: {module_name} module cost time is {avg_time:.2f} Seconds\n")
                    log_file.write(f"CMF::===== MODULE COST TEST  END =====\n")
                    log_file.flush()
                    log_file.close()
    return