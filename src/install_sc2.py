import os
import subprocess
import shutil

# Set the SMAC maps path
smac_maps = os.path.join(os.getcwd(), 'smac_maps')

# Define StarCraft II path
sc2_path = os.path.expanduser('~/StarCraftII')
print(f'SC2PATH is set to {sc2_path}')

# Check if StarCraft II is installed
if not os.path.isdir(sc2_path):
    print('StarCraftII is not installed. Installing now ...')
    subprocess.run(['wget', 'http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip'])
    subprocess.run(['unzip', '-P', 'iagreetotheeula', 'SC2.4.10.zip'])
    os.remove('SC2.4.10.zip')
else:
    print('StarCraftII is already installed.')

# Add SMAC maps
print('Adding SMAC maps.')
map_dir = os.path.join(sc2_path, 'Maps')
print(f'MAP_DIR is set to {map_dir}')

if not os.path.isdir(map_dir):
    os.makedirs(map_dir)

# Download and unzip SMAC maps
subprocess.run(['wget', 'https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip'])
subprocess.run(['unzip', 'SMAC_Maps.zip'])

# Copy maps to the StarCraft II Maps directory
shutil.copytree(smac_maps, os.path.join(map_dir, 'SMAC_Maps'), dirs_exist_ok=True)

# Clean up
os.remove('SMAC_Maps.zip') #this doesnt work

print('StarCraft II and SMAC are installed.')