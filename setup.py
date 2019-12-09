from setuptools import setup, find_packages

setup(
    name='gower',
    version='0.0.1',
    description='Python implementation of Gowers distance, pairwise between records in two data sets',
    long_description='',
    keywords=['gower', 'distance', 'matrix'],
    url='https://datainfrastructure.visualstudio.com/Data%20Science/_git/gbi_utils',
    author='Michael Yan',
    author_email='tanbingy@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.7',
    ],
    package_data={
        # If any package contains *.txt files, include them:
        #         '': ['*.sav'],
        # And include any *.dat files found in the 'data' subdirectory
        # of the 'mypkg' package, also:
        #'customer_models': ['model_objs/*.sav'],
    })