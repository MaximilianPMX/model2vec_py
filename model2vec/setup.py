from setuptools import setup, find_packages

setup(
    name='model2vec',
    version='0.1.0',
    packages=find_packages(where='model2vec'),
    package_dir={'': 'model2vec'},
    install_requires=[
        'torch', # or tensorflow
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'model2vec=model2vec.__main__:main',
        ],
    },
)
