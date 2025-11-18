from setuptools import setup, find_packages

setup(
    name='unified_pose_pipeline',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch==2.2.2',
        'torchvision==0.17.2',
        'ultralytics==8.3.228',
        'opencv-python==4.12.0.88',
        'numpy==1.26.4',
        'boxmot==15.0.10',
        'rtmlib>=0.0.6',
        'PyYAML>=6.0',
    ],
    author='Pradeep J',
    description='Unified pipeline for object tracking and 2D pose estimation',
    python_requires='>=3.8',
)