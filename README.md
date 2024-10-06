# algo_utils_hr

This package is a collection of various functions for algorithm development and data analysis.
It is primarily written in Python and uses several libraries such as numpy, scipy, and skimage for image processing and data manipulation.

## Description

The project is organized into several modules, each containing functions related to a specific area of algorithm development or data analysis. Here is a brief overview of the modules:

- `algo_typing`: This module contains type definitions used across the project.
- `data_analysis`: This module contains functions for analyzing data.
- `file_operations`: This module contains functions for file operations.
- `image_processing`: This module contains functions for image processing.
- `measurements`: This module contains functions for taking measurements from data.
- `segmentation_features`: This module contains functions for extracting features from segmented data.
- `segmentation_processing`: This module contains functions for processing segmented data.

All the functions can be imported either through the relevant module or straight from the package name `algo_utils_hr`.

## Getting Started

### Installing

This package can be installed via pip:

```bash
pip install algo_utils_hr
```

### Usage

Import the required modules and use the functions as per your requirements. For example:

```python
from algo_utils_hr import segmentation_processing

# Use the functions
segmentation_processing.get_connected_components(your_data)
```

Or you can import functions directly from `algo_utils_hr`:

```python
from algo_utils_hr import get_connected_components

# Use the function
get_connected_components(your_data)
```

## Documentation

Detailed documentation for each function is provided in the respective module files.

## License
This project is licensed under the terms of the MIT license.

## Acknowledgments
This project was developed as part of the development of algorithms for image processing and data analysis at
High-Rad LTD.

## Contact
For any queries or suggestions, please contact shalom.rochman@high-rad.com

## Authors
Shalom Rochman - https://www.linkedin.com/in/shalom-rochman-056427153/
