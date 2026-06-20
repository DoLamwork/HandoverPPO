"""Setup script for the HandoverOptimDRL package."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="handover-ppo",
    version="1.0.0",
    description="Event-triggered SMDP/PPO framework for adaptive handover.",
    author="DoLamwork",
    url="https://github.com/DoLamwork/HandoverPPO",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    include_package_data=True,
    zip_safe=False,
)
