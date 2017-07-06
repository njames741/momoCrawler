# -*- coding: utf-8 -*-

import requests
import _uniout
from bs4 import BeautifulSoup
from pprint import pprint
import re
import sys
import cStringIO
import urllib2
from PIL import Image
import pandas as pd
import csv
import os.path
import pytesseract
import time

url = "https://img3.momoshop.com.tw/expertimg/0004/856/789/TMP248_PDF_new.jpg?t=1498557587277"
file = cStringIO.StringIO(urllib2.urlopen(url).read())
img = Image.open(file)
img.show()
width, height = img.size
print width,height