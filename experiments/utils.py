import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath) if dname.startswith('events')]
    assert len(summary_iterators) == 1
    tags = set(*[si.Tags()['scalars'] for si in summary_iterators])
    
    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            out[tag].append([e.value for e in events])
    return out, steps

def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)
    df = pd.DataFrame(dict((f"{tags[i]}", np_values[i][:, 0]) for i in range(np_values.shape[0])), index=steps, columns=tags)
    df.to_csv(os.path.join(dpath, "logger.csv"))

def read_event(path):
    to_csv(path)
    return pd.read_csv(os.path.join(path, "logger.csv"), index_col=0)

def empty_dir(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))