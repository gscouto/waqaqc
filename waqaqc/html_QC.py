import datapane as dp
import matplotlib.pyplot as plt
import os
import configparser
import json

def html_l1(self):
    title = 'test'
    title = dp.HTML('<center>%s</center><hr>' % title)

    fig = plt.figure(figsize=(4, 5))

    ax = plt.plot([0, 0])

    qc_object = dp.Group(title, fig)

    layout = dp.Report(qc_object)
    layout.save('html_test.html')