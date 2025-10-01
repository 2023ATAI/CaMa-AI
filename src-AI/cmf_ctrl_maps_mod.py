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
import torch
from fortran_tensor_3D import Ftensor_3D
from fortran_tensor_2D import Ftensor_2D
from fortran_tensor_1D import Ftensor_1D

os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


class CMF_MAPS_NMLIST_MOD:
    def __init__(self,      config,    Datatype,    CC_NMLIST):
        self.device                     =           config['device']
        # --------------------------------------------------------------------------------------------------------------
        self.I2NEXTX_TyPe               =           Datatype.JPIM                            # !! POINT DOWNSTREAM  HORIZONTAL
        self.I2NEXTY_TyPe               =           Datatype.JPIM                            # !! POINT DOWNSTREAM VERTICAL
        self.I1SEQX_TyPe                =           Datatype.JPIM                            # !! 1D SEQUENCE HORIZONTAL
        self.I1SEQY_TyPe                =           Datatype.JPIM                            # !! 1D SEQUENCE VERTICAL
        self.NSEQRIV_TyPe               =           Datatype.JPIM                            # !! LENGTH OF 1D SEQUNECE FOR RIVER
        self.NSEQALL_TyPe               =           Datatype.JPIM                            # !! LENGTH OF 1D SEQUNECE FOR RIVER AND MOUTH
        self.NSEQMAX_TyPe               =           Datatype.JPIM                            # !! MAX OF NSEQALL (PARALLEL)

        self.I2VECTOR_TyPe              =           Datatype.JPIM                            # !! VECTOR INDEX
        self.I2REGION_TyPe              =           Datatype.JPIM                            # !! REGION INDEX
        self.REGIONALL                  =           CC_NMLIST.REGIONALL                      # !! REGION TOTAL
        self.REGIONTHIS                 =           CC_NMLIST.REGIONTHIS                     # !! REGION THIS CPU
        self.MPI_COMM_CAMA_TyPe         =           Datatype.JPIM                            # !! MPI COMMUNICATOR
        # --------------------------------------------------------------------------------------------------------------
        #   lat, lon
        self.D1LON_TyPe                 =           Datatype.JPRB                            # !! longitude [degree_east]
        self.D1LAT_TyPe                 =           Datatype.JPRB                            # !! latitude  [degree_north]
        # --------------------------------------------------------------------------------------------------------------
        #!*** River + Floodplain topography (map)
        self.D2GRAREA_TyPe              =           Datatype.JPRB                            # !! GRID AREA [M2]
        self.D2ELEVTN_TyPe              =           Datatype.JPRB                            # !! ELEVATION [M]
        self.D2NXTDST_TyPe              =           Datatype.JPRB                            # !! DISTANCE TO THE NEXT GRID [M]
        self.D2RIVLEN_TyPe              =           Datatype.JPRB                            # !! RIVER LENGTH [M]
        self.D2RIVWTH_TyPe              =           Datatype.JPRB                            # !! RIVER WIDTH [M]
        self.D2RIVMAN_TyPe              =           Datatype.JPRB                            # !! RIVER MANNING COEFFICIENT
        self.D2RIVHGT_TyPe              =           Datatype.JPRB                            # !! RIVER HEIGHT [M]
        self.D2FLDHGT_TyPe              =           Datatype.JPRB                            # !! FLOODPLAIN HEIGHT [M]

        self.D2GDWDLY_TyPe              =           Datatype.JPRB                            # !! Ground water delay
        self.D2ELEVSLOPE_TyPe           =           Datatype.JPRB                            # !! River bed slope
        self.I2MASK_TyPe                =           Datatype.JPIM                            # !! Mask
        # --------------------------------------------------------------------------------------------------------------
        # !*** Floodplain Topography (diagnosed)
        self.D2RIVSTOMAX_TyPe           =           Datatype.JPRB                            # !! maximum river storage [m3]
        self.D2RIVELV_TyPe              =           Datatype.JPRB                            # !! elevation of river bed [m3]
        self.D2FLDSTOMAX_TyPe           =           Datatype.JPRB                            # !! MAXIMUM FLOODPLAIN STORAGE [M3]
        self.D2FLDGRD_TyPe              =           Datatype.JPRB                            # !! FLOODPLAIN GRADIENT
        self.DFRCINC_TyPe               =           Datatype.JPRB                            # !! FLOODPLAIN FRACTION INCREMENT [-] (1/NLFP)
        # --------------------------------------------------------------------------------------------------------------
        # !*** Downstream boundary
        self.D2MEANSL_TyPe              =           Datatype.JPRB                            # !! MEAN SEA LEVEL [M]
        self.D2SEALEV_TyPe              =           Datatype.JPRB                            # !! sea level variation [m]
        self.D2DWNELV_TyPe              =           Datatype.JPRB                            # !! downstream boundary elevation [m]
        # --------------------------------------------------------------------------------------------------------------
        # !*** bifurcation channel
        self.NPTHOUT_TyPe               =           Datatype.JPIM                            # !! NUMBER OF FLOODPLAIN PATH
        self.NPTHLEV_TyPe               =           Datatype.JPIM                            # !! NUMBER OF FLOODPLAIN PATH LAYER
        self.PTH_UPST_TyPe              =           Datatype.JPIM                            # !! FLOOD PATHWAY UPSTREAM   ISEQ
        self.PTH_DOWN_TyPe              =           Datatype.JPIM                            # !! FLOOD PATHWAY DOWNSTREAM JSEQ
        self.PTH_DST_TyPe               =           Datatype.JPRB                            # !! FLOOD PATHWAY DISTANCE [m]
        self.PTH_ELV_TyPe               =           Datatype.JPRB                            # !! FLOOD PATHWAY ELEVATION [m]
        self.PTH_WTH_TyPe               =           Datatype.JPRB                            # !! FLOOD PATHWAY WIDTH [m]
        self.PTH_MAN_TyPe               =           Datatype.JPRB                            # !!  FLOOD PATHWAY Manning
        # --------------------------------------------------------------------------------------------------------------
        self.PMANRIV                    =           CC_NMLIST.PMANRIV                        # Manning coefficient for river
        self.PMANFLD                    =           CC_NMLIST.PMANFLD                        # Manning coefficient for floodplain
        # --------------------------------------------------------------------------------------------------------------
        # !*** 2. default value
        self.CNEXTXY                    =               "./nextxy.bin"
        self.CGRAREA                    =               "./ctmare.bin"
        self.CELEVTN                    =               "./elevtn.bin"
        self.CNXTDST                    =               "./nxtdst.bin"
        self.CRIVLEN                    =               "./rivlen.bin"
        self.CFLDHGT                    =               "./fldhgt.bin"

        self.CRIVWTH                    =               "./rivwth_gwdlr.bin"
        self.CRIVHGT                    =               "./rivhgt.bin"
        self.CRIVMAN                    =               "./rivman.bin"

        self.CPTHOUT                    =               "./bifprm.txt"
        self.CGDWDLY                    =               "NONE"
        self.CMEANSL                    =               "NONE"

        self.CMPIREG                    =               "NONE"

        self.LMAPCDF                    =                False
        self.CRIVCLINC                  =               "NONE"
        self.CRIVPARNC                  =               "NONE"
        self.CMEANSLNC                  =               "NONE"
        self.CMPIREGNC                  =               "NONE"
        # --------------------------------------------------------------------------------------------------------------
    def CMF_MAPS_NMLISTT(self, config,  CC_NMLIST):
        """
        ! reed setting from namelist
        ! -- Called from CMF_DRV_NMLIST
        """
        log_filename        =       config['RDIR']  +   config['LOGOUT']
        # --------------------------------------------------------------------------------------------------------------
        # !*** 1. open namelist
        with open(log_filename, 'a') as log_file:
            # Write settings to log
            log_file.write("\n!---------------------!\n")
            log_file.write(f"CMF::TIME_NMLIST: namelist OPEN in unit:   {CC_NMLIST.CSETFILE}\n")
            log_file.flush()
            log_file.close()

        # --------------------------------------------------------------------------------------------------------------
        # !*** 3. read namelist
        self.CNEXTXY                    =               config['CNEXTXY'] if 'CNEXTXY' in config  else self.CNEXTXY
        self.CGRAREA                    =               config['CGRAREA'] if 'CGRAREA' in config  else self.CGRAREA
        self.CELEVTN                    =               config['CELEVTN'] if 'CELEVTN' in config  else self.CELEVTN
        self.CNXTDST                    =               config['CNXTDST'] if 'CNXTDST' in config  else self.CNXTDST
        self.CRIVLEN                    =               config['CRIVLEN'] if 'CRIVLEN' in config  else self.CRIVLEN
        self.CFLDHGT                    =               config['CFLDHGT'] if 'CFLDHGT' in config  else self.CFLDHGT

        self.CRIVWTH                    =               config['CRIVWTH'] if 'CRIVWTH' in config  else self.CRIVWTH
        self.CRIVHGT                    =               config['CRIVHGT'] if 'CRIVHGT' in config  else self.CRIVHGT
        self.CRIVMAN                    =               config['CRIVMAN'] if 'CRIVMAN' in config  else self.CRIVMAN

        self.CPTHOUT                    =               config['CPTHOUT'] if 'CPTHOUT' in config  else self.CPTHOUT
        self.CGDWDLY                    =               config['CGDWDLY'] if 'CGDWDLY' in config  else self.CGDWDLY
        self.CMEANSL                    =               config['CMEANSL'] if 'CMEANSL' in config  else self.CMEANSL
        self.CMPIREG                    =               config['CMPIREG'] if 'CMPIREG' in config  else self.CMPIREG
        self.LMAPCDF                    =               config['LMAPCDF'] if 'LMAPCDF' in config  else self.LMAPCDF
        self.CRIVCLINC                  =               config['CRIVCLINC'] if 'CRIVCLINC' in config  else self.CRIVCLINC
        self.CRIVPARNC                  =               config['CRIVPARNC'] if 'CRIVPARNC' in config  else self.CRIVPARNC
        self.CMEANSLNC                  =               config['CMEANSLNC'] if 'CMEANSLNC' in config  else self.CMEANSLNC
        self.CMPIREGNC                  =               config['CMPIREGNC'] if 'CMPIREGNC' in config  else self.CMPIREGNC

        with open(log_filename, 'a') as log_file:
            log_file.write("=== NAMELIST, NMAP ===\n")
            log_file.write(f"LMAPCDF:   {self.LMAPCDF}\n")
            if self.LMAPCDF:
                log_file.write(f"CRIVCLINC:     {self.CRIVCLINC}\n")
                log_file.write(f"CRIVPARNC:     {self.CRIVPARNC}\n")
                if CC_NMLIST.LMEANSL:
                    log_file.write(f"CMEANSLNC: {self.CMEANSLNC}\n")
            else:
                log_file.write(f"CNEXTXY:   {self.CNEXTXY}\nCGRAREA:    {self.CGRAREA}\nELEVTN:     {self.CELEVTN}\n")
                log_file.write(f"CNXTDST:   {self.CNXTDST}\nCRIVLEN:    {self.CRIVLEN}\nCFLDHGT:    {self.CFLDHGT}\n")
                log_file.write(f"CRIVWTH:   {self.CRIVWTH}\nCRIVHGT:    {self.CRIVHGT}\nCRIVMAN:    {self.CRIVMAN}\n")
                log_file.write(f"CPTHOUT:   {self.CPTHOUT}\n")
            if CC_NMLIST.LGDWDLY:
                log_file.write(f"CGDWDLY:   {self.CGDWDLY}\n")
            if CC_NMLIST.LMEANSL:
                log_file.write(f"CMEANSL:   {self.CMEANSL}\n")
            log_file.write("CMF::MAP_NMLIST: end")
            log_file.flush()
            log_file.close()
        # --------------------------------------------------------------------------------------------------------------
    def CMF_RIVMAP_INIT(self, CC_NMLIST,         log_filename,            Datatype,         config):
        # --------------------------------------------------------------------------------------------------------------
        """
        ! read & set river network map
        ! -- call from CMF_DRV_INIT
        CONTAINS    :
                    !+ READ_MAP_BIN
                    !+ READ_MAP_CDF
                    !+ CALC_REGION
                    !+ CALC_1D_SEQ
                    !+ READ_BIFPRM
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def READ_MAP_BIN(CCNMLIST_Class, log_filename, Datatype, device):
            #   !*** read river map
            with open(log_filename, 'a') as log_file:
                log_file.write(f"RIVMAP_INIT: nextxy binary:     {self.CNEXTXY}\n")
                log_file.flush()
                log_file.close()
            with (open(self.CNEXTXY, 'rb') as f):
                I2NEXTX_temp            =           torch.frombuffer(bytearray(f.read(4 * CCNMLIST_Class.NX * CCNMLIST_Class.NY)),
                                                               dtype=self.I2NEXTX_TyPe).clone().reshape(CCNMLIST_Class.NY, CCNMLIST_Class.NX).T
                self.I2NEXTX[:CCNMLIST_Class.NX+1, :CCNMLIST_Class.NY+1]          =           I2NEXTX_temp
                I2NEXTY_temp            =           torch.frombuffer(bytearray(f.read(4 * CCNMLIST_Class.NX * CCNMLIST_Class.NY)),
                                                                dtype=self.I2NEXTY_TyPe).clone().reshape(CCNMLIST_Class.NY, CCNMLIST_Class.NX).T
                self.I2NEXTY[:CCNMLIST_Class.NX + 1, :CCNMLIST_Class.NY + 1]      =           I2NEXTY_temp
                f.close()


            if CCNMLIST_Class.LMAPEND:
                print("The 'CONV_ENDI' code in 176-th Line for cmf_ctrl_maps_mod.py is needed to improved")
                print("The 'CONV_ENDI' code in 177-th Line for cmf_ctrl_maps_mod.py is needed to improved")

            #   !*** calculate lat, lon
            if CCNMLIST_Class.WEST >= -180.0 and CCNMLIST_Class.EAST <= 360.0 and CCNMLIST_Class.SOUTH >= -180.0 and CCNMLIST_Class.NORTH <= 180.0:
                self.IX                   =          torch.arange(1, CCNMLIST_Class.NX + 1, dtype=Datatype.JPIM, device=device)
                self.D1LON[:]             =          CCNMLIST_Class.WEST + (self.IX - 0.5) * (CCNMLIST_Class.EAST - CCNMLIST_Class.WEST) / CCNMLIST_Class.NX
                self.IY                   =          torch.arange(1, CCNMLIST_Class.NY + 1, dtype=Datatype.JPIM, device=device)
                self.D1LAT[:]             =          CCNMLIST_Class.NORTH - (self.IY - 0.5) * (CCNMLIST_Class.NORTH - CCNMLIST_Class.SOUTH) / CCNMLIST_Class.NY
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CALC_REGION(CCNMLIST_Class, log_filename, device):
            #!! evenly allocate pixels to mpi nodes (updated in v4.03. MPI region given from file)
            with open(log_filename, 'a') as log_file:
                log_file.write("RIVMAP_INIT: region code\n")
                log_file.flush()
                log_file.close()

            #   !*** read MPI region map
            CCNMLIST_Class.REGIONALL        =       1
            I2REGION_temp                   =       torch.full(
                                                    (CCNMLIST_Class.NX, CCNMLIST_Class.NY), CCNMLIST_Class.IMIS,
                                                    device=device)
            self.I2REGION                   =       Ftensor_2D(I2REGION_temp, start_row=1, start_col=1)
            #   Compute the mask of grids with downstream points: [I2REGION[mask]=1]
            mask                            =       Ftensor_2D(self.I2NEXTX.raw() != CCNMLIST_Class.IMIS, start_row=1, start_col=1)
            self.I2REGION[mask]             =       1

            with open(log_filename, 'a') as log_file:
                log_file.write("RIVMAP_INIT: count number of grid in each region:\n")
                log_file.flush()
                log_file.close()
                self.REGIONGRID             =       torch.zeros ((CCNMLIST_Class.REGIONALL), dtype=self.I2REGION.raw().dtype, device=device)
                unique, counts              =       torch.unique(self.I2REGION[self.I2REGION > 0], return_counts=True)
                self.REGIONGRID[unique-1]   =       counts[0]

                self.NSEQMAX                =       torch.max(self.REGIONGRID).item()
                self.NSEQALL                =       torch.tensor(0)

                with open(log_filename, 'a') as log_file:
                    log_file.write(f"CALC_REGION: REGIONALL=      {CCNMLIST_Class.REGIONALL}\n")
                    log_file.write(f"CALC_REGION: NSEQMAX=        {self.NSEQMAX}\n")
                    log_file.write(f"CALC_REGION: NSEQALL='       {self.NSEQALL}\n")
                    log_file.flush()
                    log_file.close()
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def CALC_1D_SEQ(CC_NMLIST, Datatype, device):
            with open(log_filename, 'a') as log_file:
                log_file.write("RIVMAP_INIT: convert 2D map to 1D sequence\n")
                log_file.flush()
                log_file.close()

            NUPST_temp              =       torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY), dtype=Datatype.JPIM,device=device)
            self.NUPST              =       Ftensor_2D(NUPST_temp, start_row=1, start_col=1)
            UPNOW_temp              =       torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY), dtype=Datatype.JPIM,device=device)
            self.UPNOW              =       Ftensor_2D(UPNOW_temp, start_row=1, start_col=1)

            I1SEQX_temp             =       torch.zeros((self.NSEQMAX), dtype=Datatype.JPIM,device=device)
            self.I1SEQX             =       Ftensor_1D(I1SEQX_temp,    start_index=1)
            I1SEQY_temp             =       torch.zeros((self.NSEQMAX), dtype=Datatype.JPIM,device=device)
            self.I1SEQY             =       Ftensor_1D(I1SEQY_temp,    start_index=1)
            I1NEXT_temp             =       torch.zeros((self.NSEQMAX), dtype=Datatype.JPIM,device=device)
            self.I1NEXT             =       Ftensor_1D(I1NEXT_temp,    start_index=1)
            I2VECTOR_temp           =       torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY), dtype=Datatype.JPIM,device=device)
            self.I2VECTOR           =       Ftensor_2D(I2VECTOR_temp, start_row=1, start_col=1)
        # --------------------------------------------------------------------------------------------------------------
        # ! count number of upstream
        # 1-st Role: This is only for counting how many upstream grid cells each grid cell has (i.e., how many river channels flow into it).
        #       It does not handle topological sorting, nor does it modify the registration information.
        #       It scans all grid cells once to quickly generate the entire NUPST graph structure.
        # ! mask: grid in I2REGION is valid and grid in I2NEXTX has downstream coordinate
            mask                    =       Ftensor_2D((self.I2NEXTX.raw() > 0) & (self.I2REGION.raw() == self.REGIONTHIS),
                                               start_row=self.I2NEXTX.start_row, start_col=self.I2NEXTX.start_col)
            IX, IY                  =       mask.where(mask.raw())
            # To match the storage order of the Fortran version, sorting is performed along the IY axis during conversion
            sorted_indices          =       torch.argsort(IY, stable=True)
            IX, IY                  =       IX[sorted_indices], IY[sorted_indices]
            # Extract downstream coordinate for the current grid point.
            JX                      =       self.I2NEXTX[IX, IY]
            JY                      =       self.I2NEXTY[IX, IY]
            # NUPST: Count the total number of upstream river segments associated with the current grid point.
            values                  =       torch.ones_like(JX, dtype=torch.int32)
            self.NUPST.index_put_((JX, JY), values, accumulate=True)
        #  1-st End
        # --------------------------------------------------------------------------------------------------------------
        # ! register upmost grid in 1d sequence
        # 2-nd Role: At this point, all values in UPNOW(IX, IY) are zero.
        #            If a grid cell's NUPST equals 0, it means it has no upstream cells and can serve as a starting point for topological sorting.
        #            These points form the first layer of the sorting sequence.
        # mask                                        :     downstream coordinate.
        # NUPST == UPNOW (values are 0)               :     no upstream coordinate.
        # mask_upmost(mask and (NUPST == UPNOW))      :     upmost grid coordinate.
        # ISEQ                                        :     marks the one-dimensional array positions belonging to the upmost grids.
            mask_upmost             =       Ftensor_2D(mask & (self.NUPST.raw() == self.UPNOW.raw()), start_row=1, start_col=1)
            IX, IY = mask_upmost.where(mask_upmost.raw())
            # To match the storage order of the Fortran version, sorting is performed along the IY axis during conversion
            sorted_indices          =       torch.argsort(IY, stable=True)
            IX, IY                  =       IX[sorted_indices], IY[sorted_indices]
            ISEQ                    =       IX.size(0)
            self.I1SEQX[:ISEQ]      =       IX
            self.I1SEQY[:ISEQ]      =       IY
            # I2VECTOR: indicates the river segment number of the corresponding grid point in the flattened array
            self.I2VECTOR[IX, IY]   =       torch.arange(1, ISEQ + 1, dtype=Datatype.JPIM,device=device)

            ISEQ1, ISEQ2            =       1, ISEQ
        # 2-nd End
        # --------------------------------------------------------------------------------------------------------------
        # !! if all upstream calculated, register to 1D sequence.
        # 3-rd Role: Scan the downstream grid cells of already registered nodes.
        # Update their UPNOW values.
        # If a grid cell has all its upstream nodes registered, add it to the topological sequence.
        # Iterate layer by layer until no new nodes can be registered (AGAIN = 0).
            AGAIN = True
            while AGAIN:
                AGAIN               =       False
                JSEQ                =       ISEQ2
                for ISEQ in range(ISEQ1, ISEQ2 + 1):
                    # upmost  coordinate
                    IX              =       self.I1SEQX[ISEQ]
                    IY              =       self.I1SEQY[ISEQ]
                    # downstream coordinate for upmost
                    JX              =       self.I2NEXTX[IX, IY]
                    JY              =       self.I2NEXTY[IX, IY]

                    self.UPNOW[JX, JY]      += 1
                    #  Has both upstream and downstream — it is an “intermediate grid cell” in the middle of the process                          :     downstream coordinate.
                    #         # UPNOW == NUPST (values are 0)           :     has all its upstream nodes registered.
                    #         # I2NEXTX[JX, JY] > 0                     :     has downstream coordinate
                    if self.UPNOW[JX, JY] == self.NUPST[JX, JY] and self.I2NEXTX[JX, JY] > 0:
                        JSEQ                     += 1
                        self.I1SEQX[JSEQ]        =           JX
                        self.I1SEQY[JSEQ]        =           JY
                        self.I2VECTOR[JX, JY]    =           JSEQ
                        AGAIN = True

                ISEQ1 = ISEQ2 + 1
                ISEQ2 = JSEQ
            self.NSEQRIV            =       JSEQ            # The number of grid points with completed topological sorting
                                                            # (including intermediate and upstream points).
                                                            # !! END OF RIVER-LINK GRID

        # --------------------------------------------------------------------------------------------------------------
        # !! Inclusion of terminal grid points (i.e., those without downstream connections).
        # 4-th Role: Identify all grid points with negative downstream values (i.e., river endpoints).
        #            Append these unregistered one-dimensional sequence grid points to the existing sequence.
        #            Update I1SEQX, I1SEQY, and I2VECTOR while maintaining consistency with the existing structure.
            ISEQ                    =       self.NSEQRIV
        # mask                                        :     terminal grid point mask.
        # I2NEXTX.raw() < 0                           :     no downstream coordinate.
        # I2NEXTX ≠ IMIS                              :     no IMIS
        # I2REGION == REGIONTHIS	                  :     current model region
            mask                    =       Ftensor_2D((self.I2NEXTX.raw() < 0) & (self.I2NEXTX.raw() != CC_NMLIST.IMIS) &
                                                 (self.I2REGION.raw() == self.REGIONTHIS),start_row=1, start_col=1)
            maskIX, maskIY          =       mask.where(mask.raw())
            sorted_indices          =       torch.argsort(maskIY, stable=True)  # 按 IY 递增排序
            maskIX, maskIY          =       maskIX[sorted_indices], maskIY[sorted_indices]

            mask_count              =       maskIX.size(0)
            self.I1SEQX[ISEQ + 1:ISEQ + 1 + mask_count]         =       maskIX
            self.I1SEQY[ISEQ + 1:ISEQ + 1 + mask_count]         =       maskIY
            self.I2VECTOR[maskIX, maskIY]                       =       torch.arange(ISEQ + 1, ISEQ + 1 + mask_count, dtype=torch.int32, device=device)
            ISEQ                    +=      mask_count
            self.NSEQALL            =       ISEQ      # Record the final total number of one-dimensional river grid points, including
                                                      # (upstream, intermediate, and terminal points.)
                                                      # !! END OF RIVER-MOUTH GRID
            # --------------------------------------------------------------------------------------------------------------
            # !! Establish a one-dimensional "downstream" pointer for each grid point.
            # 5-th Role: NSEQRIV stores the total number of grid points with completed topological sorting  (including
            # upstream and intermediate points) and serves as the starting index.
            #            If a grid point has a downstream, find the identifier of the downstream grid point and establish
            # the indexing relationship.
            #           If there is no downstream (terminal point), retain the negative value as the terminal point identifier.
            JX                              =       self.I2NEXTX[self.I1SEQX.raw(), self.I1SEQY.raw()]
            JY                              =       self.I2NEXTY[self.I1SEQX.raw(), self.I1SEQY.raw()]
            mask                            =       Ftensor_1D(JX > 0, start_index=1)
            self.I1NEXT[mask.raw()]         =       self.I2VECTOR[JX[mask.raw()], JY[mask.raw()]]
            self.I1NEXT[~mask.raw()]        =       self.I2NEXTX[self.I1SEQX[~mask.raw()], self.I1SEQY[~mask.raw()]]
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def READ_BIFPARAM(Datatype, device):    #!! evenly allocate pixels to mpi nodes (not used in vcurrent version)
            #Read and initialize parameters related to river network bifurcations,
            # which are used to simulate the hydrodynamic processes of branched river channels.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"RIVMAP_INIT: Bifuraction channel:   {self.CPTHOUT}\n")
                log_file.flush()
                log_file.close()

                with open(self.CPTHOUT, 'rb') as f:
                    lines = [line.decode("utf-8").strip() for line in f.readlines()]
                    numbers = [s for s in lines[0].replace(",", " ").split() if s.lstrip("-").isdigit()]
                    self.NPTHOUT, self.NPTHLEV          =           map(int, numbers[:2])

                with open(log_filename, 'a') as log_file:
                    log_file.write(f"Bifurcation channel dimantion:   {self.NPTHOUT,    self.NPTHLEV}\n")
                    log_file.flush()
                    log_file.close()

                PTH_UPST_temp      =            torch.zeros((self.NPTHOUT), dtype=self.PTH_UPST_TyPe, device=device)
                self.PTH_UPST      =            Ftensor_1D(PTH_UPST_temp,  start_index=1)
                PTH_DOWN_temp      =            torch.zeros((self.NPTHOUT), dtype=self.PTH_DOWN_TyPe, device=device)
                self.PTH_DOWN      =            Ftensor_1D(PTH_DOWN_temp,  start_index=1)
                PTH_DST_temp       =            torch.zeros((self.NPTHOUT), dtype=self.PTH_DST_TyPe, device=device)
                self.PTH_DST       =            Ftensor_1D(PTH_DST_temp,  start_index=1)
                PTH_ELV_temp       =            torch.zeros((self.NPTHOUT, self.NPTHLEV), dtype=self.PTH_ELV_TyPe, device=device)
                self.PTH_ELV       =            Ftensor_2D(PTH_ELV_temp, start_row=1, start_col=1)
                PTH_WTH_temp       =            torch.zeros((self.NPTHOUT, self.NPTHLEV), dtype=self.PTH_WTH_TyPe, device=device)
                self.PTH_WTH       =            Ftensor_2D(PTH_WTH_temp, start_row=1, start_col=1)
                PTH_MAN_temp       =            torch.zeros((self.NPTHLEV), dtype=self.PTH_MAN_TyPe, device=device)
                self.PTH_MAN       =            Ftensor_1D(PTH_MAN_temp,  start_index=1)

                IPTH_Index           =            torch.arange(1, self.NPTHOUT + 1, device=device)
                with open(self.CPTHOUT, 'rb') as f:
                    lines           =           f.readlines()
                    data            =           torch.tensor([[float(value) for value in line.split()] for line in lines[1:]]
                                                             , dtype=Datatype.JPRB, device=device)
                    IX, IY, JX, JY                                  =       data[:, 0],   data[:, 1],  data[:, 2],  data[:, 3]
                    self.PTH_DST[:]                                 =       data[:, 4].to(dtype=self.PTH_DST_TyPe)
                    PELV_temp                                       =       data[:, 5].clone().detach().to(dtype=Datatype.JPRB, device=device)
                    self.PELV                                       =       Ftensor_1D(PELV_temp, start_index=1)
                    PDPH_temp                                       =       data[:, 6].clone().detach().to(dtype=Datatype.JPRB, device=device)
                    self.PDPH                                       =       Ftensor_1D(PDPH_temp, start_index=1)
                    self.PTH_WTH[:,:self.NPTHLEV+1]        =       data[:, 7:7 + self.NPTHLEV].to(dtype=self.PTH_WTH_TyPe)

                PTH_UPST            =           self.I2VECTOR[IX.long(), IY.long()]
                PTH_DOWN            =           self.I2VECTOR[JX.long(), JY.long()]
                self.PTH_UPST[:]    =           PTH_UPST
                self.PTH_DOWN[:]    =           PTH_DOWN
                PU_P_M_ID           =           ((self.PTH_UPST.raw() > 0) & (self.PTH_DOWN.raw() > 0)).nonzero(as_tuple=True)[0]
                for ILEV in range (1 , self.NPTHLEV+1):
                    if ILEV == 1:       # !!ILEV=1: water channel bifurcation. consider bifurcation channel depth
                        PWTH                =           self.PTH_WTH[IPTH_Index,ILEV]
                        PWTH_P_M            =           (PWTH >   0).nonzero(as_tuple=True)[0]
                        PWTH_N_M            =           (PWTH <=  0).nonzero(as_tuple=True)[0]
                        self.PTH_ELV[IPTH_Index[PWTH_P_M],ILEV]  =  self.PELV[IPTH_Index[PWTH_P_M]] - self.PDPH[IPTH_Index[PWTH_P_M]]
                        self.PTH_ELV[IPTH_Index[PWTH_N_M],ILEV]  =  1.0e20
                    else:              #  !! ILEV=2: bank top level
                        PWTH                =           self.PTH_WTH[IPTH_Index,ILEV]
                        PWTH_P_M = (PWTH > 0).nonzero(as_tuple=True)[0]
                        PWTH_N_M = (PWTH <= 0).nonzero(as_tuple=True)[0]
                        self.PTH_ELV[IPTH_Index[PWTH_P_M], ILEV] = self.PELV[IPTH_Index[PWTH_P_M]] + ILEV - 2
                        self.PTH_ELV[IPTH_Index[PWTH_N_M], ILEV] = 1.0e20
                f.close()
                self.PTH_MAN[1].fill_   (self.PMANRIV)
                self.PTH_MAN[2:].fill_  (self.PMANFLD)


                if self.NPTHOUT != PU_P_M_ID.shape[0]:
                    log_file.write(f"Bifuraction channel outside of domain. Only valid:   {PU_P_M_ID.shape[0]}\n")
                    log_file.write(f"CMF::RIVMAP_INIT: end")
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write("\n!---------------------!\n")
            log_file.write("CMF::RIVMAP_INIT: river network initialization\n")
            log_file.flush()
            log_file.close()

        #   ! *** 1. ALLOCATE ARRAYS
        self.I2NEXTX            =               torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY),
                                                            dtype=self.I2NEXTX_TyPe, device=self.device)
        self.I2NEXTX            =               Ftensor_2D      (self.I2NEXTX,      start_row=1,        start_col=1)
        self.I2NEXTY            =               torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY),
                                                            dtype=self.I2NEXTY_TyPe,device=self.device)
        self.I2NEXTY            =               Ftensor_2D      (self.I2NEXTY,      start_row=1,        start_col=1)
        self.I2REGION           =               torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY),
                                                            dtype=self.I2REGION_TyPe,device=self.device)
        self.I2REGION           =               Ftensor_2D      (self.I2REGION,      start_row=1,        start_col=1)
        self.D1LON              =               torch.zeros((CC_NMLIST.NX),
                                                            dtype=self.D1LON_TyPe,device=self.device)
        self.D1LON              =               Ftensor_1D      (self.D1LON,                            start_index=1)
        self.D1LAT              =               torch.zeros((CC_NMLIST.NY),
                                                            dtype=self.D1LAT_TyPe,device=self.device)
        self.D1LAT              =               Ftensor_1D      (self.D1LAT,                            start_index=1)

        # --------------------------------------------------------------------------------------------------------------
        #   ! *** 2a. read river network map
        with open(log_filename, 'a') as log_file:
            log_file.write("CMF::RIVMAP_INIT: read nextXY & set lat lon\n")
            log_file.flush()
            log_file.close()
        if self.LMAPCDF:
            print("The 'READ_MAP_CDF' code in 182-th Line for cmf_ctrl_maps_mod.py is needed to improved")
        else:
            READ_MAP_BIN    (CC_NMLIST,  log_filename,   Datatype,  self.device)

        #   !*** 2b. calculate river sequence & regions
        with open(log_filename, 'a') as log_file:
            log_file.write("CMF::RIVMAP_INIT: calc region\n")
            log_file.flush()
            log_file.close()
            CALC_REGION      (CC_NMLIST,  log_filename,   self.device)

        # --------------------------------------------------------------------------------------------------------------
        #   !*** 3. conversion 2D map -> 1D vector
            with open(log_filename, 'a') as log_file:
                log_file.write("CMF::RIVMAP_INIT: calculate 1d river sequence\n")
                log_file.flush()
                log_file.close()
            #   !! 2D map to 1D vector conversion. for faster calculation
            CALC_1D_SEQ    (CC_NMLIST,  Datatype,  self.device)

            with open(log_filename, 'a') as log_file:
                log_file.write(f"NSEQRIV=: {self.NSEQRIV}\n")
                log_file.write(f"NSEQALL=: {self.NSEQALL}\n")
                log_file.write(f"NSEQMAX=: {self.NSEQMAX}\n")
                log_file.flush()
                log_file.close()

            #   !*** 3c. Write Map Data
            if self.REGIONTHIS == 1:
                with open(config["RDIR"]   +   config["EXP"]    +      "mapdata.txt", "w") as f:
                    f.write(f"NX    {CC_NMLIST.NX}\n")
                    f.write(f"NY    {CC_NMLIST.NY}\n")
                    f.write(f"NLFP  {CC_NMLIST.NLFP}\n")
                    f.write(f"REGIONALL     {self.REGIONALL}\n")
                    f.write(f"NSEQMAX       {self.NSEQMAX}\n")

        # --------------------------------------------------------------------------------------------------------------
        #   !*** 4.  bifurcation channel parameters
            if CC_NMLIST.LPTHOUT:
                with open(log_filename, 'a') as log_file:
                    log_file.write("CMF::RIVMAP_INIT: read bifurcation channel setting\n")
                    log_file.flush()
                    log_file.close()
                READ_BIFPARAM       (Datatype,  self.device)

            with open(log_filename, 'a') as log_file:
                log_file.write("CMF::RIVMAP_INIT: end\n")
                log_file.flush()
                log_file.close()
        return
        # --------------------------------------------------------------------------------------------------------------
    def CMF_TOPO_INIT(self, CC_NMLIST,       log_filename,      Datatype,         CU):
        """
        ! read & set topography map
        ! -- call from CMF_DRV_INIT
        CONTAINS    :
                    !+ READ_TOPO_BIN
                    !+ READ_TOPO_CDF
                    !+ SET_FLDSTG
                    !+ SET_SLOPEMIX
        """
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def READ_TOPO_BIN(CC_NMLIST,log_filename,device,Datatype,CMF_UTILS_Class):
            import numpy as np

            R2TEMP_TyPe             =           Datatype.JPRM
        # --------------------------------------------------------------------------------------------------------------
            R2TEMP                  =           torch.zeros((CC_NMLIST.NX, CC_NMLIST.NY), dtype=R2TEMP_TyPe, device=device)
            R2TEMP                  =           Ftensor_2D(R2TEMP, start_row=1, start_col=1)


        # --------------------------------------------------------------------------------------------------------------
        # D2GRAREA:  catchment area
        # role    :  represents the catchment area upstream of each grid cell, used to quantify the contributing flow
        #            accumulation in hydrological modeling.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: unit-catchment area :   {self.CGRAREA}\n")
                log_file.flush()
                log_file.close()

            with open(self.CGRAREA, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:,:]         =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2GRAREA       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()
        # --------------------------------------------------------------------------------------------------------------
        # D2ELEVTN:  bank top elevation
        # role    :  represents the elevation of the riverbank top for each grid cell, which is used to assess potential
        #            overflow and channel capacity.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: ground elevation :   {self.CELEVTN}\n")
                log_file.flush()
                log_file.close()

            with open(self.CELEVTN, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:, :]        =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2ELEVTN       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()
        # --------------------------------------------------------------------------------------------------------------
        # D2NXTDST:  distance to next outlet
        # role    :  represents the distance from each grid cell to its next downstream outlet, used to determine flow
        #            path length in routing calculations.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: downstream distance :   {self.CNXTDST}\n")
                log_file.flush()
                log_file.close()

            with open(self.CNXTDST, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:, :]        =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2NXTDST       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()
        # --------------------------------------------------------------------------------------------------------------
        # D2RIVLEN:  river channel length
        # role    :  represents the length of the river channel within each grid cell, used to calculate flow travel time
        #            and hydraulic properties
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: river channel length :   {self.CRIVLEN}\n")
                log_file.flush()
                log_file.close()

            with open(self.CRIVLEN, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:, :]        =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2RIVLEN       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()
        # --------------------------------------------------------------------------------------------------------------
        # D2FLDHGT:  floodplain elevation profile
        # role    :  represents the elevation profile of the floodplain for each grid cell, used to simulate floodplain
        #            inundation and overbank flow dynamics.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: floodplain elevation profile :   {self.CFLDHGT}\n")
                log_file.flush()
                log_file.close()

            with open(self.CFLDHGT, 'rb') as f:
                for ILFP in range(CC_NMLIST.NLFP):
                    buffer          =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                    data_np         =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                    R2TEMP[:, :]    =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                    D2TEMP          =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                    self.D2FLDHGT[:, :, ILFP+1] = D2TEMP.raw()
                f.close()

            #   !*** river channel / groundwater parameters)
        # --------------------------------------------------------------------------------------------------------------
        # D2RIVHGT:  channel depth
        # role    :  represents the depth of the river channel in each grid cell, used to determine channel capacity and
        #            flow depth in hydrodynamic modeling.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: floodplain elevation profile :   {self.CRIVHGT}\n")
                log_file.flush()
                log_file.close()

            with open(self.CRIVHGT, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:, :]        =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2RIVHGT       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()
        # --------------------------------------------------------------------------------------------------------------
        # D2RIVWTH:  channel width
        # role    :  represents the width of the river channel in each grid cell, used to calculate flow capacity and
        #            water movement within the channel.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: river channel width :   {self.CRIVWTH}\n")
                log_file.flush()
                log_file.close()

            with open(self.CRIVWTH, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:, :]        =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2RIVWTH       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()
        # --------------------------------------------------------------------------------------------------------------
        # D2RIVMAN:  river manning coefficient
        # role    :  represents the Manning's roughness coefficient for the river channel in each grid cell,
        #            used to estimate flow resistance and velocity in hydraulic modeling.
            with open(log_filename, 'a') as log_file:
                log_file.write(f"TOPO_INIT: manning coefficient river:   {self.CRIVMAN}\n")
                log_file.flush()
                log_file.close()

            with open(self.CRIVMAN, 'rb') as f:
                buffer              =           f.read(4 * CC_NMLIST.NX * CC_NMLIST.NY)
                data_np             =           np.frombuffer(buffer, dtype=np.float32).reshape(CC_NMLIST.NY, CC_NMLIST.NX).T
                R2TEMP[:, :]        =           torch.tensor(data_np, dtype=R2TEMP_TyPe, device=device)
                self.D2RIVMAN       =           CMF_UTILS_Class.mapR2vecD(R2TEMP, self.I1SEQX, self.I1SEQY, self.NSEQMAX, device)
                f.close()

            if CC_NMLIST.LGDWDLY:
                with open(log_filename, 'a') as log_file:
                    log_file.write(f" TOPO_INIT: groundwater delay parameter:    {self.CGDWDLY}\n")
                    log_file.flush()
                    log_file.close()
                print("The 'groundwater' code in 634-th Line for cmf_ctrl_map_mod.py is needed to improved")
                print("The 'groundwater' code in 635-th Line for cmf_ctrl_map_mod.py is needed to improved")

            if CC_NMLIST.LMEANSL:
                with open(log_filename, 'a') as log_file:
                    log_file.write(f" TOPO_INIT: mean sea level:    {self.CMEANSL}\n")
                    log_file.flush()
                    log_file.close()
                print("The 'mean sea level' code in 642-th Line for cmf_ctrl_map_mod.py is needed to improved")
                print("The 'mean sea level' code in 643-th Line for cmf_ctrl_map_mod.py is needed to improved")

            return
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        def SET_FLDSTG(CC_NMLIST,device,Datatype):
            """
             Role: Calculate the distribution of storage volume distance (D2FLDSTOMAX) and water surface slope (D2FLDGRD)
             of the floodplain (water surface) under different water level elevations (layered), for use in hydrodynamic
             models to simulate the evolution of water flow over the floodplain
             -----------------------------------------------------------------------------------------------------------
        #    DSTONOW = D2RIVLEN(ISEQ,1) * ( D2RIVWTH(ISEQ,1) + DWTHINC*(DBLE(I)-0.5) ) * (D2FLDHGT(ISEQ,1,I)-DHGTPRE)
        #     Calculate the water storage distance generated by the incremental storage volume at the current layer:
        #    -----------------------------------------------------------------------------------------------------------
        #    Term                |    Description                   |      Explanation
        #    D2RIVLEN(ISEQ,1)    |    River segment length          |      Represents the length of the river segment associated
        #                        |                                  |      with the current floodplain profile.
        #    D2RIVWTH(ISEQ,1)	 |    Base channel width	        |     The initial bottom width of the river channel,
        #                        |                                  |      serving as the reference for lateral expansion.
        #    DWTHINC             |    Incremental water surface     |      Represents the horizontal expansion of the water
        #                        |    width per layer               |      surface per layer, derived from the catchment area distribution.
        #                        |    	                            |
        #    DBLE(I) - 0.5	     |    Midpoint index of the current |      Converts the integer index I to a floating-point
        #                        |    layer	                        |     number and shifts it to represent the midpoint of the layer.
        #    D2RIVWTH +          |    Mean water surface width      |      The average width of the water surface at the current layer,
        #    DWTHINC*            |    at laye                       |      accounting for lateral flood expansion.
        #   (DBLE(I) - 0.5)	     |                                  |
        #                        |                                  |
        #    D2FLDHGT(ISEQ,1,I)  |    Incremental water depth       |      The vertical difference between the current and
        #     - DHGTPRE	         |     of the current layer         |      previous floodplain elevation layers.
            """
            self.D2FLDSTOMAX                =       torch.zeros((self.NSEQALL,1,CC_NMLIST.NLFP),dtype=Datatype.JPRB, device=device)
            self.D2FLDSTOMAX                =       Ftensor_3D(self.D2FLDSTOMAX, start_depth=1, start_row=1, start_col=1)
            self.D2FLDGRD                   =       torch.zeros((self.NSEQALL,1,CC_NMLIST.NLFP),dtype=Datatype.JPRB, device=device)
            self.D2FLDGRD                   =       Ftensor_3D(self.D2FLDGRD, start_depth=1, start_row=1, start_col=1)
            self.DFRCINC                    =       torch.tensor(CC_NMLIST.NLFP, dtype=torch.float64,device=device).pow(-1)
            #!
            NQ_Index                        =       torch.arange(1, self.NSEQMAX + 1, device=device)
            self.DSTOPRE        =       self.D2RIVSTOMAX[NQ_Index, 1].clone().detach().to(dtype=Datatype.JPRB, device=device)
            self.DHGTPRE        =       torch.zeros((self.NSEQMAX), dtype=Datatype.JPRB, device=device)
            self.DWTHINC        =       ((self.D2GRAREA[NQ_Index, 1] * self.D2RIVLEN[NQ_Index, 1] ** (-1.) * self.DFRCINC).
                                         clone().detach().to(dtype=Datatype.JPRB, device=device))

            for I  in range(1,  CC_NMLIST.NLFP+1):
                self.DSTONOW                =       (self.D2RIVLEN[NQ_Index, 1]     *
                                                    (self.D2RIVWTH[NQ_Index, 1]     +   self.DWTHINC *
                                                     (torch.tensor(I,dtype=Datatype.JPRB) - 0.5)) *
                                                    (self.D2FLDHGT[NQ_Index,1,I]    -   self.DHGTPRE))
                self.D2FLDSTOMAX[NQ_Index,1,I]     =       self.DSTOPRE + self.DSTONOW
                self.D2FLDGRD[NQ_Index, 1, I]      =      (self.D2FLDHGT[NQ_Index, 1, I] - self.DHGTPRE) * self.DWTHINC ** (-1)
                self.DSTOPRE                       =       self.D2FLDSTOMAX[NQ_Index,1,I]
                self.DHGTPRE                       =       self.D2FLDHGT[NQ_Index,1,I]
            return
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n!---------------------!\n")

            log_file.write("CMF::TOPO_INIT: topography map initialization\n")
            log_file.flush()
            log_file.close()

        #   ! *** 1. ALLOCATE ARRAYS
        self.D2GRAREA        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2GRAREA_TyPe, device=self.device)
        self.D2GRAREA        =                  Ftensor_2D(self.D2GRAREA, start_row=1, start_col=1)
        self.D2ELEVTN        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2ELEVTN_TyPe, device=self.device)
        self.D2ELEVTN        =                  Ftensor_2D(self.D2ELEVTN, start_row=1, start_col=1)
        self.D2NXTDST        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2NXTDST_TyPe, device=self.device)
        self.D2NXTDST        =                  Ftensor_2D(self.D2NXTDST, start_row=1, start_col=1)
        self.D2RIVLEN        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2RIVLEN_TyPe, device=self.device)
        self.D2RIVLEN        =                  Ftensor_2D(self.D2RIVLEN, start_row=1, start_col=1)
        self.D2RIVWTH        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2RIVWTH_TyPe, device=self.device)
        self.D2RIVWTH        =                  Ftensor_2D(self.D2RIVWTH, start_row=1, start_col=1)
        self.D2RIVHGT        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2RIVHGT_TyPe, device=self.device)
        self.D2RIVHGT        =                  Ftensor_2D(self.D2RIVHGT, start_row=1, start_col=1)
        self.D2FLDHGT        =                  torch.zeros((self.NSEQMAX, 1, CC_NMLIST.NLFP), dtype=self.D2FLDHGT_TyPe, device=self.device)
        self.D2FLDHGT        =                  Ftensor_3D(self.D2FLDHGT, start_depth=1, start_row=1, start_col=1)
        self.D2RIVMAN        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2RIVMAN_TyPe, device=self.device)
        self.D2RIVMAN        =                  Ftensor_2D(self.D2RIVMAN, start_row=1, start_col=1)
        self.D2MEANSL        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2MEANSL_TyPe, device=self.device)
        self.D2MEANSL        =                  Ftensor_2D(self.D2MEANSL, start_row=1, start_col=1)
        self.D2DWNELV        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2DWNELV_TyPe, device=self.device)
        self.D2DWNELV        =                  Ftensor_2D(self.D2DWNELV, start_row=1, start_col=1)
        self.D2GDWDLY        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2GDWDLY_TyPe, device=self.device)
        self.D2GDWDLY        =                  Ftensor_2D(self.D2GDWDLY, start_row=1, start_col=1)
        #!! mask for calculation (IFS slopemix: Kinemacti Wave for Mask=1; Reservoir: dam=2, dam upstream=1)
        self.I2MASK          =                  torch.zeros((self.NSEQMAX, 1), dtype=self.I2MASK_TyPe, device=self.device)
        self.I2MASK          =                  Ftensor_2D(self.I2MASK, start_row=1, start_col=1)

        # --------------------------------------------------------------------------------------------------------------
        #   ! *** 2. Read topo map
        with open(log_filename, 'a') as log_file:
            log_file.write("CMF::TOPO_INIT: read topography maps\n")
            log_file.flush()
            log_file.close()
        if not self.LMAPCDF:
            READ_TOPO_BIN       (CC_NMLIST,log_filename,self.device,Datatype,CU)
        else:
            print("The 'READ_TOPO_CDF' code in 557-th Line for cmf_ctrl_maps_mod.py is needed to improved")
            print("The 'READ_TOPO_CDF' code in 558-th Line for cmf_ctrl_maps_mod.py is needed to improved")

    # --------------------------------------------------------------------------------------------------------------
    #   ! *** 3a. Calc Channel Parameters
        with open(log_filename, 'a') as log_file:
            log_file.write("TOPO_INIT: calc river channel parameters\n")
            log_file.flush()
            log_file.close()

        self.D2RIVSTOMAX     =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2RIVSTOMAX_TyPe, device=self.device)
        self.D2RIVSTOMAX     =                  Ftensor_2D(self.D2RIVSTOMAX, start_row=1, start_col=1)
        self.D2RIVELV        =                  torch.zeros((self.NSEQMAX, 1), dtype=self.D2RIVELV_TyPe, device=self.device)
        self.D2RIVELV        =                  Ftensor_2D(self.D2RIVELV, start_row=1, start_col=1)

        if  CC_NMLIST.LFPLAIN:
            self.D2RIVSTOMAX[:,:]       =       self.D2RIVLEN.raw() * self.D2RIVWTH.raw() * self.D2RIVHGT.raw()
        else:
            with open(log_filename, 'a') as log_file:
                log_file.write("TOPO_INIT: no floodplain (rivstomax=1.D18)\n")
                log_file.flush()
                log_file.close()
            self.D2RIVSTOMAX[:, :].fill_(1e18)
        self.D2RIVELV[:,:]               =      self.D2ELEVTN[:,:] - self.D2RIVHGT[:,:]

        #   !*** 3b. Calc Channel Parameters
        with open(log_filename, 'a') as log_file:
            log_file.write("TOPO_INIT: calc floodplain parameters\n")
            log_file.flush()
            log_file.close()

        self.D2FLDSTOMAX                =       torch.zeros((self.NSEQMAX, 1, CC_NMLIST.NLFP), dtype=self.D2FLDSTOMAX_TyPe, device=self.device)
        self.D2FLDSTOMAX                =       Ftensor_3D(self.D2FLDSTOMAX, start_depth=1, start_row=1, start_col=1)
        self.D2FLDGRD                   =       torch.zeros((self.NSEQMAX, 1, CC_NMLIST.NLFP), dtype=self.D2FLDGRD_TyPe, device=self.device)
        self.D2FLDGRD                   =       Ftensor_3D(self.D2FLDGRD, start_depth=1, start_row=1, start_col=1)
        SET_FLDSTG          (CC_NMLIST,self.device,Datatype)

        #   !*** 3c. Calc downstream boundary
        with open(log_filename, 'a') as log_file:
            log_file.write("TOPO_INIT: calc downstream boundary elevation\n")
            log_file.flush()
            log_file.close()
        self.D2DWNELV[:,:]              =       self.D2ELEVTN[:,:]
        if CC_NMLIST.LMEANSL:
            self.D2DWNELV[:,:]          =       self.D2ELEVTN[:,:] + self.D2MEANSL[:,:]
        return













