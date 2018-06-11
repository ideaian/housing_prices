#!/usr/bin/env python

from setuptools import setup
#import shutil

#srcfile = "pre-push"; dstdir = ".git/hooks"
#shutil.copy(srcfile, dstdir)

setup(
    name='predict-housing-prices',
    version='1.0',
    description='Prediction of housing prices',
    author=['Ian Derrington'],
    packages=['core','utils'],
    include_package_data=True,
    entry_points={
    }
)
