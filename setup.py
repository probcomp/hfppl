from setuptools import setup

setup(
    name='hfppl',
    version='0.1.0',    
    description='Probabilistic programming with HuggingFace Transformer models',
    url='https://github.com/probcomp/hfppl',
    author='Alex Lew',
    author_email='alexlew@mit.edu',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['torch',
                      'numpy',
                      'transformers',
                      'bitsandbytes',
                      'accelerate',
                      'sentencepiece'
                      ],

    classifiers=[
        'Programming Language :: Python :: 3.10',
    ],
)
