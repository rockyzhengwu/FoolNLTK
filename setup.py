#!/usr/bin/env python
# -*- coding:utf-8 -*-



from distutils.core import setup

setup(name = 'foolnltk',
    version = '0.1.5',
    description = 'Fool Nature Language toolkit ',
    author = 'wu.zheng',
    author_email = 'rocky.zheng314@gmail.com',
    url = 'https://github.com/rockyzhengwu/FoolNLTK',
    install_requires=['tensorflow>=1.3.0', 'numpy>=1.12.1'],
    packages = ['fool',],
    package_dir = {"fool": "fool"},
    data_files = [("fool", ["data/map.zip", "data/pos.pb", "data/seg.pb", "data/ner.pb"])]
    )
