import sys
import os
import string


chars = string.ascii_letters
outer_dir = sys.argv[1]
all_dirs = [os.path.join(outer_dir, f)
            for f in os.listdir(outer_dir)
            if not os.path.isfile(os.path.join(outer_dir, f)) and
            not f.startswith('_')]
for directory in all_dirs:
    files = [f for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    files = [f for f in files
             if f.endswith('.png') and f.split('.')[0] in chars]
    for fname in files:
        oldpath = os.path.join(directory, fname)
        new_name = ord(fname.split('.')[0])
        newpath = os.path.join(directory, new_name+'.png')
        os.rename(oldpath, newpath)
                 
