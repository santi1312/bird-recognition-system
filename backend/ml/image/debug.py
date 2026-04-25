import sys
import traceback

try:
    exec(open("ml/image/train.py").read())
except SystemExit as e:
    print("SystemExit:", e)
except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()