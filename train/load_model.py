#!/usr/bin/env python
#-*-coding:utf-8-*-

from fool.model import Model

map_file = "./datasets/demo/maps.pkl"
checkpoint_ifle = "./results/demo_seg/modle.pb"

smodel = Model(map_path=map_file, model_path=checkpoint_ifle)
tags = smodel.predict(list("北京欢迎你"))
print(tags)