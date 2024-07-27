from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Business',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='TeremokTSLib',
  version='0.0.1',
  description='Easy-to-use box ML solution for forcasting consumption',

  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  long_description_content_type="text/markdown",
  url='https://teremok.ru/',  
  author='Alexander Nikitin',
  author_email='sniknickitin@yandex.ru',
  license='MIT', 
  classifiers=classifiers,
  keywords='forcasting', 
  package_dir={"":"TeremokTSLib"},
  packages=find_packages(where="TeremokTSLib"),
  install_requires=["catboost >= 1.2.2", "pandas >= 2.1.4", "numpy >= 1.26.2", "prophet >= 1.1.5", "statsmodels >= 0.14.2", "scikit-learn >= 1.5.1"],
)