""" Entry point for WDmodel fitter"""
import sys
from .main import main

if __name__=='__main__':
    inargs = sys.argv[1:]
    main(inargs)
