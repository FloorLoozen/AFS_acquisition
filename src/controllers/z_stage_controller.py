import ctypes
import os
import time

DLL_PATH = r"C:\Program Files\Mad City Labs\NanoDrive\Labview Executable Examples\Madlib.dll"

print("Loading:", DLL_PATH)
print("Exists:", os.path.exists(DLL_PATH))

# Load DLL
nano = ctypes.WinDLL(DLL_PATH)

# Function signatures
nano.MCL_InitHandle.restype = ctypes.c_int
nano.MCL_SingleWriteZ.argtypes = [ctypes.c_double, ctypes.c_int]
nano.MCL_SingleWriteZ.restype = ctypes.c_int
nano.MCL_SingleReadZ.argtypes = [ctypes.c_int]
nano.MCL_SingleReadZ.restype = ctypes.c_double

# Connect
handle = nano.MCL_InitHandle()
print("Handle =", handle)
if handle < 0:
    raise Exception("Failed to initialize NanoDrive handle")

# Read current Z
z0 = nano.MCL_SingleReadZ(handle)
print(f"Current Z = {z0:.3f} µm")

# Move +10 µm
target = z0 + 10
print(f"Moving to {target:.3f} µm...")
nano.MCL_SingleWriteZ(target, handle)

time.sleep(0.2)

# Read again
z1 = nano.MCL_SingleReadZ(handle)
print(f"New Z = {z1:.3f} µm")
