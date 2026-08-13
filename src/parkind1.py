#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  April  24  08:42 2025
@Author: Qingliang Li: liqingliang@ccsfu.edu.cn （Email）
@Co-author1: Zhongwang Wei:  weizhw6@mail.sysu.edu.cn（Email）
@Co-author2: Kaixuan Cai:  caikx22@mails.jlu.edu.cn（Email）
@purpose: Define usual kinds for strong typing (python)
Licensed under the Apache License, Version 2.0.
"""
import torch
import  os
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'
os.environ['PYTHONWARNINGS']='ignore::RuntimeWarning'


class Parkind1:
    def __init__(self):
        """
        Class to define standard integer, real, and logical kinds for strong typing.
        """
        # Integer Kinds
        self.JPIT = torch.int8   # Equivalent to SELECTED_INT_KIND(2)
        self.JPIS = torch.int16  # Equivalent to SELECTED_INT_KIND(4)
        self.JPIM = torch.int32  # Equivalent to SELECTED_INT_KIND(9) - 4-byte integer
        self.JPIB = torch.int64  # Equivalent to SELECTED_INT_KIND(12) - 8-byte integer

        # Special integer type for address calculations (64-bit addressing optimization)
        try:
            import os
            ADDRESS64 = bool(int(os.environ.get('ADDRESS64', 0)))
        except:
            ADDRESS64 = False
        self.JPIA = self.JPIB if ADDRESS64 else self.JPIM

        # Real Kinds
        self.JPRT = torch.float16  # Equivalent to SELECTED_REAL_KIND(2,1)
        self.JPRS = torch.float32  # Equivalent to SELECTED_REAL_KIND(4,2)
        self.JPRM = torch.float32  # Equivalent to SELECTED_REAL_KIND(6,37) - 4-byte float

        # Switchable precision based on "Single Precision Mode"
        try:
            import os
            SinglePrec_CMF = bool(int(os.environ.get('SinglePrec_CMF', 0)))
        except:
            SinglePrec_CMF = False
        self.JPRB = torch.float64 if SinglePrec_CMF else torch.float64
        # Double precision for high-precision calculations
        self.JPRD = torch.float64  # Equivalent to SELECTED_REAL_KIND(13,300) - 8-byte double-precision float

        # Logical Kind for RTTOV....
        self.JPLM = self.JPIM  # Equivalent to JPIM in Fortran (logical type)


