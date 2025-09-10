# MuJoCo Blender Importer

This tool allows you to import MuJoCo XML models and animations into Blender. It supports importing mesh geometries, body hierarchies, and qpos-based animations.

## Prerequisites

- Blender 2.8+
- Python packages:
  - mujoco
  - numpy
  - tqdm

## Usage Instructions

1. Install Blender

2. Press `Ctrl+Shift+P` to open the command palette, type `start` and select `Run Script`.

3. **Important**: The script must be run from within the Python script editor in VSCODE

4. Make sure your XML model file and qpos data file match correctly:
   - The single XML file should define the full model structure
   - Export the full XML in Mujoco using the `Save xml` in the top left corner. 

## File Requirements

1. MuJoCo XML Model File:
   - Mesh paths should be correctly specified relative to the XML file
   - Change mesh dir `Geometry` to `/home/micha/Desktop/mj_blender/MS-Human-700-Internal/models/Geometry`
   - Change asset dir `Asset/marble.png` to `/home/micha/Desktop/mj_blender/MS-Human-700-Internal/models/Asset/marble.png` (2 places)
   - Example: `mjmodel_locomotion.xml`

2. QPOS Data File:
   - Should be a text file with `english comma-separated` (,) values
   - Each line represents one frame of animation
   - Number of values per line must match the model's degrees of freedom
   - Example: `test.txt`

## Script Configuration

The main script parameters are at the bottom of the file:

```python
xml_file_path = "/path/to/your/model.xml"
qpos_path = "/path/to/your/qpos.txt"
```

Update these paths to point to your model and animation data files.

## Features

- Imports mesh geometries with correct transformations
- Creates hierarchical body structure
- Supports both Euler and quaternion rotations
- Imports sites (optional, controlled by site_radius parameter)
- Creates animation keyframes from qpos data
- Organizes objects into collections:
  - TENDON: For tendon visualization
  - OTHER: For cameras and lights
  - SUPPORT: For additional support objects

## Common Issues

1. **Incorrect Animation**: 
   - Check that your qpos data matches the XML model's degrees of freedom
   - Verify the qpos data format (comma-separated values)
   - Ensure the XML file paths are correct

2. **Missing Meshes**:
   - Verify mesh paths in the XML file
   - Check that mesh files exist in the specified locations
   - Supported formats: .stl and .obj

3. **Script Errors**:
   - Make sure to run the script from within Blender's script editor
   - Check that all required Python packages are installed
   - Verify file paths are correct and accessible

## Best Practices

1. Always keep your XML model and qpos data files in sync
2. Use absolute paths to avoid path resolution issues
3. Organize your files in a clear directory structure
4. Back up your Blender file before running the import script
5. Clear existing objects/animations before re-importing if needed

## Notes

- The script creates a hierarchical structure that matches the MuJoCo model
- Animation data is stored as keyframes in Blender
- Tendon visualization is supported with adjustable thickness
- Camera and light objects are automatically organized into the OTHER collection
- Default cube objects are automatically removed
