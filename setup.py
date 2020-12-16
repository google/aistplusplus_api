import setuptools

INSTALL_REQUIREMENTS = [
    'absl-py', 'numpy', 'opencv-python']

setuptools.setup(
    name='aist_plusplus_api',
    # url='https://github.com/google/aist_plusplus_api',
    description='API for supporting AIST++ Dataset.',
    version='0.0.3',
    author='Ruilong Li',
    author_email='ruilongli94@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIREMENTS
)
