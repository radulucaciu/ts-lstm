from setuptools import setup, find_packages
import os

DESCRIPTION = "Some out-of-the-box lstm-based time series models"
if os.path.exists('README.md'):
    with open('README.md', encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
else:
    LONG_DESCRIPTION = DESCRIPTION

setup(name='lymboy-lstm',
      version='v1.6',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      url='https://github.com/lymboy/lymboy-lstm',
      author='lymboy.com',
      author_email='liusairo@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'keras',
          'scikit-learn',
          'plotly'
      ],

      zip_safe=True)
