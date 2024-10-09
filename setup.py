from setuptools import setup, find_packages

setup(
    name='algo_utils_hr',
    version='1.1.0',
    description='A collection of utility functions for algorithm development and data analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Shalom Rochman',
    author_email='shalom.rochman@high-rad.com',
    url='https://github.com/High-Rad/algo_utils.git',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        "jsonschema~=4.21.1",
        "nibabel~=5.2.1",
        "numpy~=1.26.3",
        "pandas~=2.2.2",
        "networkx~=3.1",
        "scikit-image~=0.24.0",
        "scipy~=1.14.0",
        "tqdm~=4.65.0",
        "xlsxwriter~=3.1.9",
        "medpy~=0.5.2",
    ]
)
