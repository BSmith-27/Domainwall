from setuptools import setup

setup(
    name='distributed_svpg',
    version='0.01',
    packages=['distributed_svpg'],
    install_requires=['tensorflow', 'mpi4py', 'numpy'],
    url='https://code.ornl.gov/ai/rl/svpg',
    license='MIT',
    include_package_data=True

)
