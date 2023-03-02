from setuptools import setup, find_packages
import os

DESCRIPTION = "Some out-of-the-box lstm-based time series models"  # 关于该包的剪短描述
if os.path.exists('README.md'):  # 如果需要，可以加入一段较长的描述，比如读取 README.md，该段长描述会直接显示在 PyPI 的页面上
    with open('README.md', encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
else:
    LONG_DESCRIPTION = DESCRIPTION

setup(name='lymboy-lstm',
      version='v1.6',
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      url='https://github.com/lymboy',
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
