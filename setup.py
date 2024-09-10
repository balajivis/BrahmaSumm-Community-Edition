from setuptools import setup, find_packages

# Function to read requirements.txt
def parse_requirements(filename):
    """Load requirements from a requirements.txt file."""
    with open(filename, "r") as f:
        return f.read().splitlines()

setup(
    name="BrahmaSumm",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt")
)