import os
import sys



def _append_run_path():
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    os.environ['DYLD_LIBRARY_PATH'] = os.path.join(base_dir)
    os.environ['LD_LIBRARY_PATH'] = os.path.join(base_dir)

_append_run_path()