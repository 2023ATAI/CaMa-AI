#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose:  read CaMa-Flood Model configulations from namelist ("input_flood.nam" as default) (python)
Licensed under the Apache License, Version 2.0.

* CONTAINS:
! -- CMF_CONFIG_NAMELIST  : read namelist for CaMa-Flood
! -- CMFConfigCheck     : check config conflict
"""
import  os
import torch

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


class CMF_CTRL_NMLIST_MOD:
    def __init__(self ,config,    Datatype):
        # --------------------------------------------------------------------------------------------------------------
        # *** 1. Basic simulation run version
        self.device         =       config["device"]
        self.MODTTEST       =       config["MODTTEST"] if "MODTTEST" in config  else False             # true: for testing time cost  for each module
        self.PDSTMTH        =       torch.tensor(config["PDSTMTH"], dtype=Datatype.JPRB,device=self.device)
        self.CSETFILE       =       config["CSETFILE"] if "CSETFILE" in config  else 'input_cmf.nam'
        self.LINPCDF        =       config["LINPCDF"]  if "LINPCDF"  in config  else False             # true: netCDF output, .FALSE. plain binary output
        # * Defaults
        self.LADPSTP        =       config["LADPSTP"]  if "LADPSTP"  in config  else True              # true: use adaptive time step
        self.LFPLAIN        =       config["LFPLAIN"]  if "LFPLAIN"  in config  else True              # true: consider floodplain (false: only river channel)
        self.LKINE          =       config["LKINE"]    if "LKINE"    in config  else False             # true: use kinematic wave
        self.LFLDOUT        =       config["LFLDOUT"]  if "LFLDOUT"  in config  else True              # true: floodplain flow (high-water channel flow) active
        self.LPTHOUT        =       config["LPTHOUT"]  if "LPTHOUT"  in config  else False             # true: activate bifurcation scheme
        self.LDAMOUT        =       config["LDAMOUT"]  if "LDAMOUT"  in config  else False             # true: activate dam operation
        self.LLEVEE         =       config["LLEVEE"]   if "LLEVEE"   in config  else False             # true: activate levee scheme (under development)
        self.LSEDOUT        =       config["LSEDOUT"]  if "LSEDOUT"  in config  else False             # true: activate sediment transport (under development)
        self.LTRACE         =       config["LTRACE"]   if "LTRACE"   in config  else False             # true: activate tracer (under development)
        self.LOUTINS        =       config["LOUTINS"]  if "LOUTINS"  in config  else False             # true: diagnose instantaneous discharge
        self.TESTIT         =       0
        # === This part is used by ECMWF
        self.LROSPLIT       =       config["LROSPLIT"] if "LROSPLIT" in config  else False             # true: input if surface (Qs) and sub-surface (Qsb) runoff
        self.LWEVAP         =       config["LWEVAP"]   if "LWEVAP"   in config  else False             # true: input evaporation to extract from river
        self.LWEVAPFIX      =       config["LWEVAPFIX"]if "LWEVAPFIX"in config  else False             # true: water balance closure extracting water from evaporation when available
        self.LGDWDLY        =       config["LGDWDLY"]  if "LGDWDLY"  in config  else False             # true: activate groundwater reservoir and delay
        self.LSLPMIX        =       config["LSLPMIX"]  if "LSLPMIX"  in config  else False             # true: activate mixed kinematic and local inertia based on slope
        self.LWEXTRACTRIV   =       config["LWEXTRACTRIV"]if "LWEXTRACTRIV" in config  else False      # true: also extract water from rivers
        self.LSLOPEMOUTH    =       config["LSLOPEMOUTH"] if "LSLOPEMOUTH"  in config  else False      # true: prescribe water level slope == elevation slope on river mouth

        # *** Dynamic Sea Level
        self.LMEANSL        =       config["LMEANSL"]  if "LMEANSL"  in config  else False             # true: boundary condition for mean sea level
        self.LSEALEV        =       config["LSEALEV"]  if "LSEALEV"  in config  else False             # true: boundary condition for variable sea level

        # *** Restart & Output
        self.LRESTART       =       config["LRESTART"] if "LRESTART" in config  else False             # true: initial condition from restart file
        self.LSTOONLY       =       config["LSTOONLY"] if "LSTOONLY" in config  else False             # true: storage only restart (mainly for data assimilation)
        self.LOUTPUT        =       config["LOUTPUT"]  if "LOUTPUT"  in config  else True              # true: use standard output (to file)
        self.LOUTINI        =       config["LOUTINI"]  if "LOUTINI"  in config  else False             # true: output initial storage (netCDF only)

        self.LGRIDMAP       =       config["LGRIDMAP"] if "LGRIDMAP" in config  else True              # true: for standard XY gridded 2D map
        self.LLEAPYR        =       config["LLEAPYR"]  if "LLEAPYR"  in config  else True              # true: neglect leap year (Feb29 skipped)
        self.LMAPEND        =       config["LMAPEND"]  if "LMAPEND"  in config  else False             # true: for map data endian conversion
        self.LBITSAFE       =       config["LBITSAFE"] if "LBITSAFE" in config  else False             # true: for Bit Identical (not used from v410, set in Mkinclude)
        self.LSTG_ES        =       config["LSTG_ES"]  if "LSTG_ES"  in config  else False             # true: for Vector Processor optimization (CMF_OPT_FLDSTG_ES)
        # --------------------------------------------------------------------------------------------------------------
        # *** 2. Set Model Dimension & Time
        # defaults (from namelist)
        self.CDIMINFO       =       config["CDIMINFO"] if "CDIMINFO" in config else "None"
        self.DT             =       config["DT"]       if "DT"       in config else 24*60*60          # dt = 1 day (automatically set by adaptive time step) 24 * 60 * 60
        self.IFRQ_INP       =       config["IFRQ_INP"] if "IFRQ_INP" in config else 24                # daily (24h) input

        # ============================
        # *** Change Section
        self.DTIN           =       self.IFRQ_INP * 60 * 60         # hour -> second

        # ============================
        # *** Default values (from diminfo)
        self.NX             =       1440                            # 15-minute resolution
        self.NY             =       720
        self.NLFP           =       10                              # 10 floodplain layers
        self.NXIN           =       360                             # 1 degree input
        self.NYIN           =       180
        self.INPN           =       1                               # maximum number of input grids corresponding to one CaMa-Flood grid
        # west, east, north, south edges of the domain
        self.WEST           =       torch.tensor    (-180.0,          dtype=Datatype.JPRB,  device=self.device)
        self.EAST           =       torch.tensor    (180.0,      dtype=Datatype.JPRB,  device=self.device)
        self.NORTH          =       torch.tensor    (90.0,       dtype=Datatype.JPRB,  device=self.device)
        self.SOUTH          =       torch.tensor    (-90.0,           dtype=Datatype.JPRB,  device=self.device)
        self.REGIONALL      =       1
        self.REGIONTHIS     =       1
        ##============================
        #   #*** 1h. Output Settings
        self.IFRQ_OUT       =       24                              # output frequency: [1,2,3,...,24] hour

        if self.CDIMINFO.strip().upper() != "NONE":
            print(f"CMF::CONFIG_NMLIST: read DIMINFO {self.CDIMINFO}")
            # Open the CDIMINFO file and read its contents
            with open(self.CDIMINFO, 'r') as file:
                # Read key parameters from the file
                self.NX = int(file.readline().split("!!")[0].strip())           # Grid size in X direction
                self.NY = int(file.readline().split("!!")[0].strip())           # Grid size in Y direction
                self.NLFP = int(file.readline().split("!!")[0].strip())         # Number of land surface points
                self.NXIN = int(file.readline().split("!!")[0].strip())         # Input grid size in X direction
                self.NYIN = int(file.readline().split("!!")[0].strip())         # Input grid size in Y direction
                self.INPN = int(file.readline().split("!!")[0].strip())         # Input point count
                file.readline()                                                 # Skip one line

                if self.LGRIDMAP:
                    self.WEST = float(file.readline().split("!!")[0].strip())   # Western boundary
                    self.EAST = float(file.readline().split("!!")[0].strip())   # Eastern boundary
                    self.NORTH = float(file.readline().split("!!")[0].strip())  # Northern boundary
                    self.SOUTH = float(file.readline().split("!!")[0].strip())  # Southern boundary
                file.close()
        # --------------------------------------------------------------------------------------------------------------
        # *** 3. set PARAM: parameters
        self.PMANRIV        =     torch.tensor    (config["PMANRIV"] if "PMANRIV" in config  else 0.03,
                                                   dtype=Datatype.JPRB,  device=self.device)        # Manning coefficient for river
        self.PMANFLD        =     torch.tensor    (config["PMANFLD"] if "PMANFLD" in config  else 0.1,
                                                   dtype=Datatype.JPRB,  device=self.device)        # Manning coefficient for floodplain
        self.PGRV           =     torch.tensor    (config["PGRV"]   if "PGRV" in config  else 9.8,
                                                   dtype=Datatype.JPRB,  device=self.device)        # Gravity acceleration (m/s²)
        self.PDSTMTH        =     torch.tensor    (config["PDSTMTH"]if "PDSTMTH" in config  else 10000,
                                                   dtype=Datatype.JPRB,  device=self.device)        # Downstream distance at river mouth [m]
        self.PCADP          =     torch.tensor    (config["PCADP"]  if "PCADP" in config  else 0.7,
                                                   dtype=Datatype.JPRB,  device=self.device)        # CFL coefficient
        self.PMINSLP        =     config["PMINSLP"]  if "PMINSLP" in config  else 1.E-5                                                                           # Minimum slope (kinematic wave)

        self.IMIS           =     torch.tensor    (-9999,            dtype=Datatype.JPIM,  device=self.device)
        self.RMIS           =     1.E20
        self.DMIS           =     torch.tensor    (1.E20,       dtype=Datatype.JPRM,  device=self.device)

        self.CSUFBIN        =       '.bin'                                      # Binary file extension
        self.CSUFVEC        =       '.vec'                                      # Vector file extension
        self.CSUFPTH        =       '.pth'                                      # Path file extension
        self.CSUFCDF        =       '.nc'                                       # NetCDF file extension
        # --------------------------------------------------------------------------------------------------------------
    def log_settings(self, config):
        if not os.path.exists(config['RDIR']):
            os.makedirs(config['RDIR'])
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        if os.path.exists   (log_filename):
            os.remove       (log_filename)

        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write("\n!--------------------\n")
            log_file.write(f"CMF::CONFIG_NMLIST: namelist opened:       {self.CSETFILE}\n")
            log_file.write("\n=== NAMELIST, NRUNVER ===\n")
            log_file.write(f"Cost for each module:                      {self.MODTTEST}\n")
            log_file.write(f"device                                     {self.device}\n")
            log_file.write(f"LADPSTP                                    {self.LADPSTP}\n")
            log_file.write(f"LFPLAIN                                    {self.LFPLAIN}\n")
            log_file.write(f"LKINE                                      {self.LKINE}\n")
            log_file.write(f"LFLDOUT                                    {self.LFLDOUT}\n")
            log_file.write(f"LPTHOUT                                    {self.LPTHOUT}\n")
            log_file.write(f"LDAMOUT                                    {self.LDAMOUT}\n")
            log_file.write(f"LLEVEE                                     {self.LLEVEE}\n")
            log_file.write(f"LSEDOUT                                    {self.LSEDOUT}\n")
            log_file.write(f"LTRACE                                     {self.LTRACE}\n")
            log_file.write(f"LOUTINS                                    {self.LOUTINS}\n")
            log_file.write("\n")
            log_file.write(f"LROSPLIT                                   {self.LROSPLIT}\n")
            log_file.write(f"LWEVAP                                     {self.LWEVAP}\n")
            log_file.write(f"LWEVAPFIX                                  {self.LWEVAPFIX}\n")
            log_file.write(f"LWEXTRACTRIV                               {self.LWEXTRACTRIV}\n")
            log_file.write(f"LGDWDLY                                    {self.LGDWDLY}\n")
            log_file.write(f"LSLPMIX                                    {self.LSLPMIX}\n")
            log_file.write(f"LSLOPEMOUTH                                {self.LSLOPEMOUTH}\n")
            log_file.write("\n")
            log_file.write(f"LMEANSL:                                   {self.LMEANSL}\n")
            log_file.write(f"LSEALEV:                                   {self.LSEALEV}\n")
            log_file.write("\n")
            log_file.write(f"LRESTART                                   {self.LRESTART}\n")
            log_file.write(f"LSTOONLY                                   {self.LSTOONLY}\n")
            log_file.write(f"LOUTPUT                                    {self.LOUTPUT}\n")
            log_file.write(f"LOUTINI                                    {self.LOUTINI}\n")
            log_file.write("\n")
            log_file.write(f"LGRIDMAP                                   {self.LGRIDMAP}\n")
            log_file.write(f"LLEAPYR                                    {self.LLEAPYR}\n")
            log_file.write(f"LMAPEND                                    {self.LMAPEND}\n")
            log_file.write(f"LBITSAFE                                   {self.LBITSAFE}\n")
            log_file.write(f"LSTG_ES                                    {self.LSTG_ES}\n")
        # --------------------------------------------------------------------------------------------------------------
            # Write model dimension and time settings to the log file
            log_file.write("\n=== NAMELIST, NCONF ===\n")
            log_file.write(f"CDIMINFO                                   {self.CDIMINFO}\n")
            log_file.write(f"DT                                         {self.DT}\n")
            log_file.write(f"DTIN                                       {self.DTIN}\n")
            log_file.write(f"IFRQ_INP                                   {self.IFRQ_INP}\n")
            log_file.write("\n")
            #!* value from CDIMINFO
            if self.CDIMINFO.strip().upper() != "NONE":
                log_file.write(f"CMF::CONFIG_NMLIST: read DIMINFO       {self.CDIMINFO}\n")
            log_file.write("\n=== DIMINFO ===\n")
            log_file.write(f"NX,NY,NLFP                                 {self.NX},  {self.NY},  {self.NLFP}\n")
            log_file.write(f"NXIN,NYIN,INPN                             {self.NXIN},    {self.NYIN},    {self.INPN}\n")
            if self.LGRIDMAP:
                log_file.write(f"WEST,EAST,NORTH,SOUTH                  {self.WEST},    {self.EAST},    {self.NORTH},   {self.SOUTH}\n")

            # Write model parameters to the log file
            log_file.write("\n=== NAMELIST, NPARAM ===\n")
            log_file.write(f"PMANRIV                                    {self.PMANRIV}\n")
            log_file.write(f"PMANFLD                                    {self.PMANFLD}\n")
            log_file.write(f"PGRV                                       {self.PGRV}\n")
            log_file.write(f"PDSTMTH                                    {self.PDSTMTH}\n")
            log_file.write(f"PCADP                                      {self.PCADP}\n")
            log_file.write(f"PMINSLP                                    {self.PMINSLP}\n")
            log_file.write("\n")
            log_file.write(f"IMIS                                       {self.IMIS}\n")
            log_file.write(f"RMIS                                       {self.RMIS}\n")
            log_file.write(f"DMIS                                       {self.DMIS}\n")
            log_file.write("\n")
            log_file.write(f"CSUFBIN                                    {self.CSUFBIN.strip()}\n")
            log_file.write(f"CSUFVEC                                    {self.CSUFVEC.strip()}\n")
            log_file.write(f"CSUFPTH                                    {self.CSUFPTH.strip()}\n")
            log_file.write(f"CSUFCDF                                    {self.CSUFCDF.strip()}\n")
            # Simulate file closure in Python
            log_file.write("CMF::CONFIG_NMLIST: end \n")
            log_file.write("--------------------!")
            log_file.write("\n")
            if self.REGIONALL >= 2:
                log_file.write(f"REGIONTHIS  {self.REGIONTHIS.strip()}\n")              # Regional output for MPI run
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
    def CMF_CONFIG_CHECK(self,log_filename):

        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write(f"CMF::CONFIG_CHECK: check setting conflicts\n")

            #!*** 1. check for time step
            if self.DT < 60 or int(self.DT) % 60 != 0:
                log_file.write(f"DT={self.DT}\n ")
                log_file.write(f"DT should be multiple of 60. CaMa-Flood controls time by MINUTE")
                log_file.write(f"stop")
                raise ValueError("Stop: DT should be a multiple of 60.")

            if int(self.DTIN) % int(self.DT) != 0:
                log_file.write(f"DTIN, DT={self.DTIN, self.DT}\n ")
                log_file.write(f"DTIN should be multiple of DT")
                log_file.write(f"stop")
                raise ValueError("Stop: DTIN should be a multiple of DT.")

            if self.LSEALEV and int(self.DTSL) % int(self.DT) != 0:
                log_file.write(f"DTSL, DT=={self.DTSL, self.DT}\n ")
                log_file.write(f"DTSL should be multiple of DT")
                log_file.write(f"stop")
                raise ValueError("Stop: DTSL should be a multiple of DT.")

            # Check for physics options consistency
            if not self.LFPLAIN and not self.LKINE:
                log_file.write(f"LFPLAIN=.false. & LKINE=.false.")
                log_file.write(f"CAUTION: NO FLOODPLAIN OPTION reccomended to be used with kinematic wave (LKINE=.true.)")

            if self.LKINE and self.LADPSTP:
                log_file.write(f"LKINE=.true. & LADPSTP=.true.")
                log_file.write(f"adaptive time step reccoomended only with local inertial equation (LKINE=.false.)")
                log_file.write(f"Set appropriate fixed time step for Kinematic Wave")

            if self.LKINE and self.LPTHOUT:
                log_file.write(f"LKINE=.true. & LPATHOUT=.true.")
                log_file.write(f"bifurcation channel flow only available with local inertial equation (LKINE=.false.)")
                log_file.write(f"Set appropriate fixed time step for Kinematic Wave")
                raise ValueError("Stop: LKINE=.true. & LPATHOUT=.true.")

            if self.LGDWDLY and not self.LROSPLIT:
                log_file.write(f"LGDWDLY=true and LROSPLIT=false")
                log_file.write(f"Ground water reservoir can only be active when runoff splitting is om")

            if self.LWEVAPFIX and not self.LWEVAP:
                log_file.write(f"LWEVAPFIX=true and LWEVAP=false")
                log_file.write(f"LWEVAPFIX can only be active if LWEVAP is active")

            if self.LWEXTRACTRIV and not self.LWEVAP:
                log_file.write(f"LWEXTRACTRIV=true and LWEVAP=false")
                log_file.write(f"LWEXTRACTRIV can only be active if LWEVAP is active")

                log_file.write("CMF::CONFIG_CHECK: end\n")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------