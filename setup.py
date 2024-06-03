from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirement(file_path):
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path, encoding='utf-8') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='Student-performance-indicator',
    version='0.0.1', 
    author='Khaled Zebibi',
    author_email='khaledzebibi@gmail.com', 
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)