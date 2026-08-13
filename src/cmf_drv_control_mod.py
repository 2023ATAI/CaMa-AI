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
    from cmf_ctrl_maps_mod import CMF_MAPS_NMLIST_MOD
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

    CM_NMLIST                    =       CMF_MAPS_NMLIST_MOD             (config,    Datatype,  CC_NMLIST)
    CM_NMLIST. CMF_MAPS_NMLISTT                                         (config,    CC_NMLIST)
    # --------------------------------------------------------------------------------------------------------------
    #!*** 2. read namelist for each module
    CF_NMLIST                    =       CMF_FORCING_NMLIST_MOD          (config,    Datatype,  CC_NMLIST)
    CT_NMLIST                    =       CF_NMLIST.CMF_FORCING_NMLIST    (config,  CT_NMLIST,  CC_NMLIST)

    if CC_NMLIST.LSEALEV:
        raise NotImplementedError("LSEALEV is not supported in CaMa-PyTorch v1.0.")

    CR_NMLIST                     =       CMF_RESTART_NMLIST_MOD          ()
    CR_NMLIST.CMF_RESTART_NMLIST                                          (config, CC_NMLIST)

    if CC_NMLIST.LDAMOUT:
        raise NotImplementedError("LDAMOUT is not supported in CaMa-PyTorch v1.0.")

    if CC_NMLIST.LSEALEV:
        raise NotImplementedError("LSEALEV is not supported in CaMa-PyTorch v1.0.")

    if CC_NMLIST.LOUTPUT:
        CO_NMLIST               =       CMF_OUTPUT_NMLIST_MOD           (config, Datatype)
        CO_NMLIST.CMF_OUTPUT_NMLIST                                     (config, CC_NMLIST)

    if CC_NMLIST.LSEDOUT:
        raise NotImplementedError("LSEDOUT is not supported in CaMa-PyTorch v1.0.")

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

    return (CC_NMLIST,      CT_NMLIST,    CM_NMLIST,      CF_NMLIST,   CR_NMLIST,       CO_NMLIST)
    # ------------------------------------------------------------------------------------------------------------------
    #!*** 1b. INITIALIZATION
def CMF_DRV_INIT(CC_NMLIST,             CT_NMLIST,              CM_NMLIST,      CF_NMLIST,          CR_NMLIST,
                 CO_NMLIST,             config,                 Datatype,                                               ):
    """
    ! Initialize CaMa-Flood
    ! -- Called from CMF_DRV_INIT
    """
    from cmf_ctrl_vars_mod import CMF_CTRL_VARS_MOD
    import cmf_ctrl_physics_mod
    from cmf_utils_mod import CMF_UTILS_MOD
    log_filename = config['RDIR'] + config['LOGOUT']
    CU                           =       CMF_UTILS_MOD                   (Datatype,  CC_NMLIST, CM_NMLIST)
    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n******************************!\n")
        log_file.write(f"CMF::DRV_INIT: initialization start\n")
        log_file.flush()
        log_file.close()

    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DRV_INIT: (1) Set Time\n")
        log_file.flush()
        log_file.close()

    # 1a. Set time related
    CC_NMLIST                           =          CT_NMLIST.CMF_TIME_INIT(CC_NMLIST,        log_filename,        CU)

    # ------------------------------------------------------------------------------------------------------------------
    with open(log_filename, 'a') as log_file:
        # Write settings to log
        log_file.write(f"CMF::DRV_INIT: (2) Set River Map & Topography\n")
        log_file.write("\n!---------------------!\n")

    # 2a. Read input river map
    CM_NMLIST.CMF_RIVMAP_INIT          (CC_NMLIST,         log_filename,            Datatype,         config)
    CU                           =       CMF_UTILS_MOD                   (Datatype,  CC_NMLIST, CM_NMLIST)
    # 2b. Set topography
    CM_NMLIST.CMF_TOPO_INIT            (CC_NMLIST,       log_filename,      Datatype,         CU)

    # 2c. Optional levee scheme initialization
    if CC_NMLIST.LLEVEE:
        raise NotImplementedError("LLEVEE is not supported in CaMa-PyTorch v1.0.")

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
        raise NotImplementedError("LSEALEV is not supported in CaMa-PyTorch v1.0.")
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
        raise NotImplementedError("LDAMOUT is not supported in CaMa-PyTorch v1.0.")

    #  *** 4e. Optional sediment initialization
    if CC_NMLIST.LSEDOUT:
        raise NotImplementedError("LSEDOUT is not supported in CaMa-PyTorch v1.0.")

    # ------------------------------------------------------------------------------------------------------------------
     #!** v4.03 CALC_FLDSTG moved to the top of CTRL_PHYSICS for restart configuration (Hatono & Yamazaki)

    #!*** 5 reconstruct previous t-step flow (if needed)
    if CC_NMLIST.LRESTART and CC_NMLIST.LSTOONLY:
        raise NotImplementedError("Storage-only restart flow reconstruction is not supported in CaMa-PyTorch v1.0.")

    # !*** 5b save initial storage if LOUTINI specified
    if CC_NMLIST.LOUTINI and CC_NMLIST.LOUTPUT:
        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::DRV_INIT: (5b) write initial condition\n")
            log_file.flush()
            log_file.close()
        raise NotImplementedError("Initial-condition output is not supported in CaMa-PyTorch v1.0.")

    # ------------------------------------------------------------------------------------------------------------------.

    with open(log_filename, 'a') as log_file:
        log_file.write(f"CMF::DRV_INIT: initialization finished:\n")
        log_file.write(f"CMF::DRV_INIT: end\n")
        log_file.write(f"***********************************\n")
        log_file.flush()
        log_file.close()

    return      (CC_NMLIST,             CT_NMLIST,             CM_NMLIST,       CF_NMLIST,          CR_NMLIST,
                 CO_NMLIST,             CC_VAR)

def CMF_DRV_END(config,CC_NMLIST,CF_NMLIST,CM_NMLIST,CO_NMLIST,CT_NMLIST):
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
        raise NotImplementedError("LSEALEV is not supported in CaMa-PyTorch v1.0.")
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\n!******************************!\n")
        log_file.write(f"CMF::DRV_END: simulation finished\n")
        log_file.write(f"CMF::DRV_END: close logfile\n")
        log_file.write(f"CMF::===== CALCULATION END =====\n")
        log_file.flush()
        log_file.close()
    return
