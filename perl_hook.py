
import os
import sys
import shutil

def hook(hook_api):
    perl5lib = os.environ.get('PERL5LIB', '')
    if perl5lib:
        for path in perl5lib.split(':'):
            if os.path.exists(path):
                hook_api.add_datas([(path, os.path.join('perl5lib', os.path.basename(path)))])



if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

perl5lib = os.path.join(base_path, 'uroman', 'lib')
#os.environ['PERL5LIB'] = f"{perl5lib}:{os.environ.get('PERL5LIB', '')}"
os.environ['PERL5LIB'] = perl5lib
print(f"Setting PERL5LIB to: {perl5lib}")