from setuptools import setup, find_packages

setup(
    name='satquant',
    version='0.2.2',
    description='Satellite Imagery Quantization Library with Focus Calibration',
    author='Oskar Andrukiewicz',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'opencv-python'
    ],
)
