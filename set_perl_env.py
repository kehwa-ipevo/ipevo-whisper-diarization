import os
import sys

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

perl5lib = os.path.join(base_path, 'perl5lib')
os.environ['PERL5LIB'] = perl5lib