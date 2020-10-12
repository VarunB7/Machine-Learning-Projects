# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:59:17 2020

@author: hp
"""

from PIL import Image

PngFormat = Image.open(r'C:\Users\hp\Desktop\Julybill1.png')
PdfFormat = PngFormat.convert('RGB')
PdfFormat.save(r'C:\Users\hp\Desktop\Julybill1.pdf')