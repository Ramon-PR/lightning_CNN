# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:13:21 2023

@author: Ramón Pozuelo
"""

# from tensorboard.backend.event_processing import event_accumulator

# ea = event_accumulator.EventAccumulator('events.out.tfevents.x.ip-x-x-x-x',
#    size_guidance={ # see below regarding this argument
#          event_accumulator.COMPRESSED_HISTOGRAMS: 500,
#          event_accumulator.IMAGES: 4,
#          event_accumulator.AUDIO: 4,
#          event_accumulator.SCALARS: 0,
#          event_accumulator.HISTOGRAMS: 1,
#          })


from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


# %%

name = "events.out.tfevents.1683490303.erebos.2102233.0"
scalars = ["train_loss", "val_loss"]
a = parse_tensorboard(name, scalars)

# %%

import matplotlib.pyplot as plt
plt.plot(a["train_loss"].step, [a["train_loss"].value, a["val_loss"].value])
plt.show()