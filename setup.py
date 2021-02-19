"""
Set up gym_physx
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gym_physx',
    version='0.0.1',
    author="Ingmar Schubert",
    author_email="mail@ingmarschubert.com",
    description="OpenAI-Gym Robotics Manipulation Environment Based on PhysX Simulator and KOMO planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ischubert/gym-physx",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'gym'],
    extras_require={
        "testing": ['stable-baselines3', 'matplotlib']
    },
    include_package_data=True,
    data_files=[("", [
        "gym_physx/envs/config_data/pushing.g",
        "gym_physx/envs/config_data/pushing.json"
    ])]
)
