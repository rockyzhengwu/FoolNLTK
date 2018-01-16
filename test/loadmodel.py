#!/usr/bin/env python
#-*-coding:utf-8-*-



import fool

map_file = "../datasets/demo/maps.pkl"
checkpoint_ifle = "../results/demo_seg/modle.pb"

smodel = fool.load_model(map_file=map_file, model_file=checkpoint_ifle)
tags = smodel.predict(["北京欢迎你", "你在哪里"])
print(tags)
