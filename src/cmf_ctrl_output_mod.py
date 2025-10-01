#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  MControl CaMa-Flood map/topography data (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_MAPS_NMLIST   : configuration from namelist
! -- CMF_RIVMAP_INIT  : read & set river network map
! -- CMF_TOPO_INIT    : read & set topography
"""
import  os
import re
import torch
from fortran_tensor_2D import Ftensor_2D
from netCDF4 import Dataset

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


class VarOut_Class:
    def __init__(self):
        self.CVNAME: str = ''  # ! output variable name
        self.CVLNAME: str = ''  # ! output variable long name
        self.CVUNITS: str = ''  # ! output units
        self.CFILE: str = ''  # ! output full path file name
        self.RECL: int = 0  # ! output binary output file ID
        self.BINID: int = 0  # ! output netCDF output file ID
        self.NCID: int = 0  # ! output netCDF output variable ID
        self.VARID: int = 0  # ! output netCDF time   variable ID
        self.TIMID: int = 0  # !  Current time record for writting
        self.IRECNC: int = 0
class CMF_OUTPUT_NMLIST_MOD:
    def __init__(self, config, Datatype):
        self.device                     =           config['device']
        self.COUTDIR                    =           "./"                            #    OUTPUT DIRECTORY
        self.CVARSOUT                   =           "outflw,storge,rivdph"          #    Comma-separated list of output variables to save
        self.COUTTAG                    =           "_cmf"                          #    Output Tag Name for each experiment
        #!
        self.LOUTVEC                    =           False                           #    TRUE FOR VECTORIAL OUTPUT, FALSE FOR NX,NY OUTPUT
        self.LOUTCDF                    =           False                           #    true for netcdf outptu false for binary
        self.NDLEVEL                    =           0                               #    NETCDF DEFLATION LEVEL
        self.IFRQ_OUT                   =           24                              #    !! daily (24h) output

        # !
        self.LOUTTXT                    =           False                           #    TRUE FOR Text output for some gauges
        self.CGAUTXT                    =           "None"                          #    List of Gauges (ID, IX, IY)
        #!
        # !*** local variables
        self.NVARS                      =           torch.tensor(100, dtype=Datatype.JPIM, device=self.device)    #    actual   output var number
        self.NVARSOUT                   =           0
        self.IRECOUT                    =           0                               #    Output file irec

        # !*** TYPE for output file
        # !     TYPE TVAROUT
        self.CVNAME                     =            "None"              #    output variable name
        self.CVLNAME                    =            "None"              #    output variable long name
        self.CVUNITS                    =            "None"              #    output units
        self.CFILE                      =            "None"              # output full path file name
        self.BINID_TyPe                 =           Datatype.JPIM        #    output binary output file ID
        self.NCID_TyPe                  =           Datatype.JPIM        #    output netCDF output file ID
        self.VARID_TyPe                 =           Datatype.JPIM        #    output netCDF output variable ID
        self.TIMID_TyPe                 =           Datatype.JPIM        #    output netCDF time   variable ID
        self.IRECNC_TyPe                =           Datatype.JPIM        #    Current time record for writting
        # --------------------------------------------------------------------------------------------------------------
    def CMF_OUTPUT_NMLIST(self, config, CC_NMLIST):
        """
        ! reed setting from namelist
        ! -- Called from CMF_DRV_NMLIST
        """
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        # --------------------------------------------------------------------------------------------------------------
        # !*** 1. open namelist
        with open(log_filename, 'a') as log_file:
            log_file.write("\n!---------------------!\n")
            log_file.write(f"CMF::OUTPUT_NMLIST: namelist OPEN in unit: {CC_NMLIST.CSETFILE}\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        #   !*** 3. read namelist
        self.COUTDIR                    =           config['COUTDIR']       if 'COUTDIR'  in config   else self.COUTDIR
        self.CVARSOUT                   =           config['CVARSOUT']      if 'CVARSOUT' in config   else self.CVARSOUT
        self.COUTTAG                    =           config['COUTTAG']       if 'COUTTAG'  in config   else self.COUTTAG
        #!
        self.LOUTVEC                    =           config['LOUTVEC']       if 'LOUTVEC'  in config   else self.LOUTVEC
        self.LOUTCDF                    =           config['LOUTCDF']       if 'LOUTCDF'  in config   else self.LOUTCDF
        self.NDLEVEL                    =           config['NDLEVEL']       if 'NDLEVEL'  in config   else self.NDLEVEL
        self.IFRQ_OUT                   =           config['IFRQ_OUT']      if 'IFRQ_OUT' in config   else self.IFRQ_OUT

        # !
        self.LOUTTXT                    =           config['LOUTTXT']       if 'LOUTTXT' in config   else self.LOUTTXT
        self.CGAUTXT                    =           config['CGAUTXT']       if 'CGAUTXT' in config   else self.CGAUTXT
        # --------------------------------------------------------------------------------------------------------------
        if CC_NMLIST.CSETFILE != "NONE":
            with open(CC_NMLIST.CSETFILE, 'r') as NSETFILE:
                NSETFILE.seek(0)
                NSIMTIME = {}
                for line in NSETFILE:
                    line = line.strip()
                    if "=" in line:
                        key, value = map(str.strip, line.split("=", 1))
                        NSIMTIME[key] = value
                self.CVARSOUT               =           re.findall(r"^(\S+)", NSIMTIME["CVARSOUT"], re.MULTILINE)[0]
                self.COUTTAG                =           re.findall(r"^(\S+)", NSIMTIME["COUTTAG"], re.MULTILINE)[0]
                self.COUTTAG                =           self.COUTTAG.strip().replace('"', '')
                self.COUTTAG                =           self.COUTTAG
            NSETFILE.close()
        with open(log_filename, 'a') as log_file:
            log_file.write("=== NAMELIST, NOUTPUT  ===\n")
            log_file.write(f"COUTDIR:       {self.COUTDIR.strip()}\n")
            log_file.write(f"CVARSOUT:      {self.CVARSOUT.strip()}\n")
            log_file.write(f"COUTTAG:       {self.COUTTAG.strip()}\n")

            log_file.write(f"LOUTCDF:       {self.LOUTCDF}\n")
            if self.LOUTCDF:
                log_file.write(f"NDLEVEL:       {self.NDLEVEL}\n")
            if self.LOUTVEC:
                log_file.write(f"LOUTVEC:       {self.LOUTVEC}\n")
            log_file.write(f"IFRQ_OUT:      {CC_NMLIST.IFRQ_OUT}\n")

            log_file.write(f"LOUTTXT:       {self.LOUTTXT}\n")
            log_file.write(f"CGAUTXT:       {self.CGAUTXT}\n")
            log_file.write("CMF::OUTPUT_NMLIST: end\n")
            log_file.flush()
            log_file.close()

        # --------------------------------------------------------------------------------------------------------------

    def CMF_OUTPUT_INIT(self, CC_NMLIST,          log_filename,            CM_NMLIST,           CT_NMLIST,      config):
        from typing import List
        from random import randint
        """
        ! Initialize output module (create/open files)
        ! -- Called from CMF_DRV_INIT
        CONTAINS:
        !==========================================================
        !+ CREATE_OUTBIN
        !+ CREATE_OUTCDF
        !==========================================================
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CREATE_OUTBIN(JF,CM_NMLIST,CC_NMLIST):
            if self.VAROUT[JF].CVNAME.strip() == 'pthflw':  # bifurcation channel
                if CM_NMLIST.REGIONTHIS == 1:
                    self.VAROUT[JF].CFILE        =       self.COUTDIR.strip() + self.VAROUT[JF].CVNAME.strip() + self.COUTTAG.strip() + CC_NMLIST.CSUFPTH.strip()
                    self.VAROUT[JF].RECL         =       4 * CM_NMLIST.NPTHOUT * CM_NMLIST.NPTHLEV
                    self.VAROUT[JF].BINID        =       open(self.VAROUT[JF].CFILE, 'wb')
                    with open(log_filename, 'a') as log_file:
                        log_file.write("output file opened in unit: " + self.VAROUT[JF].CFILE + ", " + str(self.VAROUT[JF].BINID) + "\n")
                        log_file.flush()
                        log_file.close()
            elif self.LOUTVEC:  # 1D land only output
                self.VAROUT[JF].CFILE            =       self.COUTDIR.strip() + self.VAROUT[JF].CVNAME.strip() + self.COUTTAG.strip() + CC_NMLIST.CSUFVEC.strip()
                self.VAROUT[JF].RECL             =       4 * CM_NMLIST.NSEQMAX
                self.VAROUT[JF].BINID            =       open(self.VAROUT[JF].CFILE, 'wb')
                with open(log_filename, 'a') as log_file:
                    log_file.write("output file opened in unit: " + self.VAROUT[JF].CFILE + ", " + str(self.VAROUT[JF].BINID) + "\n")
                    log_file.flush()
                    log_file.close()

            else:  # 2D default map output
                if CM_NMLIST.REGIONTHIS == 1:
                    self.VAROUT[JF].CFILE        =       self.COUTDIR.strip() + self.VAROUT[JF].CVNAME.strip() + self.COUTTAG.strip() + CC_NMLIST.CSUFBIN.strip()
                    with open(log_filename, 'a') as log_file:
                        log_file.write("  -- " + self.VAROUT[JF].CFILE + "\n")
                        log_file.flush()
                        log_file.close()
                    self.VAROUT[JF].RECL         =       4 * CC_NMLIST.NX * CC_NMLIST.NY
                    self.VAROUT[JF].BINID        =       open(self.VAROUT[JF].CFILE, 'wb')
                    with open(log_filename, 'a') as log_file:
                        log_file.write("output file opened in unit: " + self.VAROUT[JF].CFILE + ", " + str(self.VAROUT[JF].BINID) + "\n")
                        log_file.flush()
                        log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CREATE_OUTCDF(JF,CM_NMLIST,CC_NMLIST, config):
            #   !============
            self.VAROUT[JF].IRECNC       =   1        #   ! initialize record current writting record to 1
            self.VAROUT[JF].CFILE        =   str(f"{self.COUTDIR.strip()} o_{self.VAROUT[JF].CVNAME.strip()}{self.COUTTAG.strip()}{CC_NMLIST.CSUFCDF}")
        #   ! Create file
            NCFILE                       =   Dataset(self.VAROUT[JF].CFILE, 'w', format='NETCDF4')
            #   !=== set dimension ===
            NCFILE.createDimension('lat', CC_NMLIST.NY)
            NCFILE.createDimension('lon', CC_NMLIST.NX)
            NCFILE.createDimension      ('time', None)
            data_var = NCFILE.createVariable(self.VAROUT[JF].CVNAME, 'f4',('time', 'lat','lon'),
                                         zlib=True, complevel=self.NDLEVEL, fill_value=CC_NMLIST.RMIS)

            lat                     =   NCFILE.createVariable('lat', 'f4', ('lat',))
            lat.long_name           =   "latitude"
            lat.units               =   "degrees_north"

            lon                     =   NCFILE.createVariable('lon', 'f4', ('lon',))
            lon.long_name           =   "longitude"
            lon.units               =   "degrees_east"

            time                    =   NCFILE.createVariable('time', 'f8', ('time',))
            time.long_name          =   'time'
            time.units              =   self.CTIME


            data_var.long_name      =   ('long_name', self.VAROUT[JF].CVLNAME)
            data_var.units          =   ('units', self.VAROUT[JF].CVUNITS)

            NCFILE.variables['lon'][:]  = CM_NMLIST.D1LON.raw().cpu().numpy()
            NCFILE.variables['lat'][:]  = CM_NMLIST.D1LAT.raw().cpu().numpy()
            NCFILE.close()


            with open(log_filename, 'a') as log_file:
                log_file.write(f"CFILE: {self.VAROUT[JF].CFILE},   CVAR: {self.VAROUT[JF].CVNAME},    "f"CLNAME: {self.VAROUT[JF].CVLNAME},     CUNITS: {self.VAROUT[JF].CVUNITS}\n")
                # log_file.write(f"OPEN IN UNIT: {NCFILE}\n")
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write("\n!---------------------!\n")

            log_file.write(f" CMF::OUTPUT_INIT: check output variables\n")
            log_file.flush()
            log_file.close()

    #   !! Start by finding out # of output variables
        self.CVNAMES = [var.strip() for var in self.CVARSOUT.split(',') if var.strip()]
        self.NVARSOUT = len(self.CVNAMES)

        if self.NVARSOUT == 0:
            with open(log_filename, 'a') as log_file:
                log_file.write(f" CMF::OUTPUT_INIT: No output files will be produced!\n")
                log_file.flush()
                log_file.close()


        self.CTIME = f"seconds since {CT_NMLIST.ISYYYY:04d}-{CT_NMLIST.ISMM:02d}-{CT_NMLIST.ISDD:02d} {CT_NMLIST.ISHOUR:02d}:{CT_NMLIST.ISMIN:02d}\n"
        with open(log_filename, 'a') as log_file:
            log_file.write(self.CTIME)
            log_file.flush()
            log_file.close()

        self.VAROUT: List[VarOut_Class] = [VarOut_Class() for _ in range(self.NVARSOUT)]
        #       !* Loop on variables and create files
        for JF in range(self.NVARSOUT):
            with open(log_filename, 'a') as log_file:
                log_file.write(f"Creating output for variable: {self.CVNAMES[JF].strip()}\n")
                log_file.flush()
                log_file.close()
            varname = self.CVNAMES[JF].strip('"').strip("'")
            self.VAROUT[JF].CVNAME = str(varname)

            if varname == 'rivout':
                self.VAROUT[JF].CVLNAME = 'river discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname == 'rivsto':
                self.VAROUT[JF].CVLNAME = 'river storage'
                self.VAROUT[JF].CVUNITS = 'm3'
            elif varname == 'rivdph':
                self.VAROUT[JF].CVLNAME = 'river depth'
                self.VAROUT[JF].CVUNITS = 'm'
            elif varname == 'rivvel':
                self.VAROUT[JF].CVLNAME = 'river velocity'
                self.VAROUT[JF].CVUNITS = 'm/s'

            elif varname == 'fldout':
                self.VAROUT[JF].CVLNAME = 'floodplain discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname == 'fldsto':
                self.VAROUT[JF].CVLNAME = 'floodplain storage'
                self.VAROUT[JF].CVUNITS = 'm3'
            elif varname == 'flddph':
                self.VAROUT[JF].CVLNAME = 'floodplain depth'
                self.VAROUT[JF].CVUNITS = 'm'
            elif varname == 'fldfrc':
                self.VAROUT[JF].CVLNAME = 'flooded fraction'
                self.VAROUT[JF].CVUNITS = '0-1'
            elif varname == 'fldare':
                self.VAROUT[JF].CVLNAME = 'flooded area'
                self.VAROUT[JF].CVUNITS = 'm2'

            elif varname == 'sfcelv':
                self.VAROUT[JF].CVLNAME = 'water surface elevation'
                self.VAROUT[JF].CVUNITS = 'm'
            elif varname in ['totout', 'outflw']:           #    !! comparability for previous output name
                self.VAROUT[JF].CVLNAME = 'discharge (river+floodplain)'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname in ['totsto', 'storge']:           #    !! comparability for previous output name
                self.VAROUT[JF].CVLNAME = 'total storage (river+floodplain)'
                self.VAROUT[JF].CVUNITS = 'm3'

            elif varname == 'pthflw':
                self.VAROUT[JF].CVLNAME = 'bifurcation channel discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname == 'pthout':
                self.VAROUT[JF].CVLNAME = 'net bifurcation discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'

            elif varname == 'maxsto':
                self.VAROUT[JF].CVLNAME = 'daily maximum storage'
                self.VAROUT[JF].CVUNITS = 'm3'
            elif varname == 'maxflw':
                self.VAROUT[JF].CVLNAME = 'daily maximum discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname == 'maxdph':
                self.VAROUT[JF].CVLNAME = 'daily maximum river depth'
                self.VAROUT[JF].CVUNITS = 'm'

            elif varname == 'runoff':
                self.VAROUT[JF].CVLNAME = 'Surface runoff'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname == 'runoffsub':
                self.VAROUT[JF].CVLNAME = 'sub-surface runoff'
                self.VAROUT[JF].CVUNITS = 'm3/s'

            elif varname == 'damsto':       #   !!! added
                self.VAROUT[JF].CVLNAME = 'reservoir storage'
                self.VAROUT[JF].CVUNITS = 'm3'
            elif varname == 'daminf':       #   !!! added
                self.VAROUT[JF].CVLNAME = 'reservoir inflow'
                self.VAROUT[JF].CVUNITS = 'm3/s'

            elif varname == 'levsto':       #   !!! added
                self.VAROUT[JF].CVLNAME = 'protected area storage'
                self.VAROUT[JF].CVUNITS = 'm3'
            elif varname == 'levdph':       #   !!! added
                self.VAROUT[JF].CVLNAME = 'protected area depth'
                self.VAROUT[JF].CVUNITS = 'm'

            elif varname in ['gdwsto', 'gwsto']:
                self.VAROUT[JF].CVLNAME = 'ground water storage'
                self.VAROUT[JF].CVUNITS = 'm3'
            elif varname == 'gwout':
                self.VAROUT[JF].CVLNAME = 'ground water discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'

            elif varname == 'wevap':
                self.VAROUT[JF].CVLNAME = 'water evaporation'
                self.VAROUT[JF].CVUNITS = 'm3/s'
            elif varname == 'outins':
                self.VAROUT[JF].CVLNAME = 'instantaneous discharge'
                self.VAROUT[JF].CVUNITS = 'm3/s'

            else:
                with open(log_filename, 'a') as log_file:
                    log_file.write(f"{varname} Not defined in CMF_CREATE_OUTCDF_MOD\n")
                    log_file.flush()
                    log_file.close()
                continue  # Skip to next JF

            self.VAROUT[JF].BINID        =       randint(10000, 99999)

            # 输出方式选择
            if self.LOUTCDF:
                if CM_NMLIST.REGIONTHIS == 1:
                    CREATE_OUTCDF(JF,CM_NMLIST,CC_NMLIST,config)
            else:
                CREATE_OUTBIN(JF,CM_NMLIST,CC_NMLIST)
        self.IRECOUT =  0           # ! Initialize Output record to 1 (shared in netcdf & binary)
        return

    def CMF_OUTPUT_WRITE(self, log_filename, CC_VARS, CT_NMLIST, CC_NMLIST, Datatype, device,CM_NMLIST,CU):
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def WRTE_OUTPTH(IFN,IREC,R2OUTDA):
            offset = (IREC - 1) * R2OUTDA.raw().numel() * 4
            IFN.seek(offset)
            IFN.write(R2OUTDA.raw().to(torch.float32).cpu().numpy().tobytes())
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def WRTE_OUTBIN(IFN,IREC,R2OUTDA):
            offset = (IREC - 1) * R2OUTDA.raw().numel() * 4
            IFN.seek(offset)
            # Datatype.JPRB
            IFN.write(R2OUTDA.raw().to(torch.float32).cpu().numpy().tobytes())
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def WRTE_OUTCDF(CT_NMLIST, R2OUT, JF):

            NCFILE = Dataset(self.VAROUT[JF].CFILE, 'a', format='NETCDF4')
            # XTIME:    ! seconds since start of the run !
            XTIME                                =       ((CT_NMLIST.KMINNEXT - CT_NMLIST.KMINSTART) * 60).to(dtype=Datatype.JPRB)  #    !! for netCDF

            NCFILE.variables['time'][self.VAROUT[JF].IRECNC-1]                             =       XTIME.cpu().numpy()  # equivalent to NF90_PUT_VAR for time
            NCFILE.variables[self.VAROUT[JF].CVNAME][self.VAROUT[JF].IRECNC-1, :, :]       =       R2OUT.raw().T.cpu().numpy()
            NCFILE.close()
            #   ! update IREC
            self.VAROUT[JF].IRECNC      =        self.VAROUT[JF].IRECNC  +  1
            #   ! Comment out this as it slows down significantly the writting in the cray
            #   !CALL NCERROR( NF90_SYNC(VAROUT(JF)%NCID) )
            return
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        R1POUT                  =               torch.zeros((CM_NMLIST.NPTHOUT, CM_NMLIST.NPTHLEV),dtype=Datatype.JPRM,device=device)
        R1POUT                  =               Ftensor_2D      (R1POUT,      start_row=1,        start_col=1)
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write(f"!******************************!\n")
            log_file.flush()
            log_file.close()

        #!*** 0. check date:hour with output frequency
        if (CT_NMLIST.JHOUR % CC_NMLIST.IFRQ_OUT == 0) and (CT_NMLIST.JMIN == 0):       #! JHOUR: end of time step , NFPPH: output frequency (hour)

        # !*** 1. update IREC & calc average variable
            self.IRECOUT = self.IRECOUT + 1
            with open(log_filename, 'a') as log_file:
                log_file.write(f"CMF::OUTPUT_WRITE: write at time:  {CT_NMLIST.JYYYYMMDD},{CT_NMLIST.JHHMM}, {self.IRECOUT}\n")
                log_file.flush()
                log_file.close()

        #!*** 2. check variable name & allocate data to pointer DVEC
            for JF in range(self.NVARSOUT):
                CVNAME          =       self.VAROUT[JF].CVNAME
                if CVNAME ==        'rivsto':
                    D2VEC       =                   CC_VARS.P2RIVSTO        #   !! convert Double to Single precision when using SinglePrecisionMode
                elif CVNAME ==      'fldsto':
                    D2VEC       =                   CC_VARS.P2FLDSTO

                elif CVNAME ==      'rivout':
                    D2VEC       =                   CC_VARS.D2RIVOUT_aAVG
                elif CVNAME ==      'rivdph':
                    D2VEC       =                   CC_VARS.D2RIVDPH
                elif CVNAME ==      'rivvel':
                    D2VEC       =                   CC_VARS.D2RIVVEL_aAVG
                elif CVNAME ==      'fldout':
                    D2VEC       =                   CC_VARS.D2FLDOUT_aAVG

                elif CVNAME ==      'flddph':
                    D2VEC       =                   CC_VARS.D2FLDDPH
                elif CVNAME ==      'fldfrc':
                    D2VEC       =                   CC_VARS.D2FLDFRC
                elif CVNAME ==      'fldare':
                    D2VEC       =                   CC_VARS.D2FLDARE
                elif CVNAME ==      'sfcelv':
                    D2VEC       =                   CC_VARS.D2SFCELV

                elif CVNAME in      ['totout', 'outflw']:
                    D2VEC       =                   CC_VARS.D2OUTFLW_aAVG        #   !!  compatibility for previous file name
                elif CVNAME in      ['totsto', 'storge']:
                    D2VEC       =                   CC_VARS.D2STORGE            #   !!  compatibility for previous file name

                elif CVNAME ==      'pthout':
                    if not CC_NMLIST.LPTHOUT:
                        continue
                    D2VEC       =                   CC_VARS.D2PTHOUT_aAVG
                elif CVNAME ==      'pthflw':
                    if not CC_NMLIST.LPTHOUT:
                        continue
                elif CVNAME ==      'maxflw':
                    D2VEC       =                   CC_VARS.D2OUTFLW_aMAX
                elif CVNAME ==      'maxdph':
                    D2VEC       =                   CC_VARS.D2RIVDPH_aMAX
                elif CVNAME == '    maxsto':
                    D2VEC       =                   CC_VARS.D2STORGE_aMAX

                elif CVNAME ==      'outins':
                    if not CC_NMLIST.LOUTINS:
                        continue
                    D2VEC       =                   CC_VARS.D2OUTINS

                elif CVNAME in      ['gwsto', 'gdwsto']:
                    if not CC_NMLIST.LGDWDLY:
                        continue
                    D2VEC       =                   CC_VARS.P2GDWSTO
                elif CVNAME in      ['gwout', 'gdwrtn']:
                    if not CC_NMLIST.LGDWDLY:
                        continue
                    D2VEC       =                   CC_VARS.D2GDWRTN_aAVG

                elif CVNAME in      ['runoff', 'rofsfc']:      #   !!  compatibility for previous file name
                    D2VEC       =                   CC_VARS.D2RUNOFF_aAVG
                elif CVNAME in      ['runoffsub', 'rofsub']:
                    if not CC_NMLIST.LROSPLIT:
                        continue
                    D2VEC       =                   CC_VARS.D2ROFSUB_aAVG
                elif CVNAME ==      'wevap':
                    if not CC_NMLIST.LWEVAP:
                        continue
                    D2VEC       =                   CC_VARS.D2WEVAPEX_aAVG

                elif CVNAME ==      'damsto':           #   !!! added
                    if not CC_NMLIST.LDAMOUT:
                        continue
                    D2VEC       =                   CC_VARS.P2DAMSTO
                # elif CVNAME == 'daminf':
                #     if not CC_NMLIST.LDAMOUT:
                #         continue
                #     D2VEC       =                   d2daminf_avg

                elif CVNAME ==      'levsto':       #   !!! added
                    if not CC_NMLIST.LLEVEE:
                        continue
                    D2VEC       =                   CC_VARS.P2LEVSTO

                if CT_NMLIST.KSTEP == 0 and CC_NMLIST.LOUTINI:      #   !! write storage only when LOUTINI specified
                    if not self.LOUTCDF:
                        continue
                    if (self.VAROUT[JF].CVNAME          !=      'rivsto'    and
                            self.VAROUT[JF].CVNAME      !=      'fldsto'    and
                            self.VAROUT[JF].CVNAME      !=      'gwsto'):
                        continue

                #   !! convert 1Dvector to 2Dmap
                if self.VAROUT[JF].CVNAME               !=      'pthflw':
                    R2OUT                               =        CU.vecD2mapR(D2VEC, CM_NMLIST.I1SEQX, CM_NMLIST.I1SEQY, CM_NMLIST.NSEQMAX, device)
                else:
                    if not CC_NMLIST.LPTHOUT:
                        continue
                    R1POUT[:, :]     =                  CC_VARS.D1PTHFLW_AVG[:, :].to(dtype=torch.float64)

                #   !*** 3. write D2VEC to output file
                if self.LOUTCDF:
                    if CM_NMLIST.REGIONTHIS == 1:
                        WRTE_OUTCDF(CT_NMLIST, R2OUT, JF)
                else:
                    if self.VAROUT[JF].CVNAME  == "pthflw":
                        if CM_NMLIST.REGIONTHIS == 1:
                            WRTE_OUTPTH(self.VAROUT[JF].BINID, self.IRECOUT, R1POUT)
                    else:
                        if self.LOUTVEC:
                            WRTE_OUTCDF(CT_NMLIST, R2OUT, JF)
                        else:
                            if CM_NMLIST.REGIONTHIS == 1:
                                WRTE_OUTBIN(self.VAROUT[JF].BINID, self.IRECOUT, R2OUT)
            with open(log_filename, 'a') as log_file:
                log_file.write("CMF::OUTPUT_WRITE: end\n")
                log_file.flush()
                log_file.close()
    def CMF_OUTPUT_END(self, log_filename, CM_NMLIST):
        """
        ! Finalize output module (close files)
        ! -- Called from CMF_DRV_END
        """
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n!---------------------!\n")
            log_file.write(f"CCMF::OUTPUT_END: finalize output modul\n")
            log_file.flush()
            log_file.close()

        if CM_NMLIST.REGIONTHIS==1:
            if self.LOUTCDF:
                for JF in range(self.NVARSOUT):
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"Output netcdf output unit closed:  {self.VAROUT[JF].CVNAME}\n")
                        log_file.flush()
                        log_file.close()
            else:       #   !! binary output
                for JF in range(self.NVARSOUT):
                    self.VAROUT[JF].BINID.close()
                    with open(log_filename, 'a') as log_file:
                        log_file.write(f"Output binary output unit closed:  {self.VAROUT[JF].BINID}\n")
                        log_file.flush()
                        log_file.close()

        with open(log_filename, 'a') as log_file:
            log_file.write(f"CMF::OUTPUT_END: end\n")
            log_file.flush()
            log_file.close()

        return