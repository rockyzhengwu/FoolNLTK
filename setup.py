#!/usr/bin/env python
# -*- coding:utf-8 -*-



from distutils.core import setup

setup(name = 'foolnltk',
    version = '0.0.8',
    description = 'Fool Nature Language toolkit ',
    author = 'wu.zheng',
    author_email = 'rocky.zheng314@gmail.com',
    url = 'https://github.com/rockyzhengwu/FoolNLTK',
    packages = ['fool'],
    package_dir = {"fool": "fool"},
    data_files = [("fool", ["data/maps.pkl", "data/ner_pos.pb", "data/seg.pb", "data/pos.pb"])]
    )
