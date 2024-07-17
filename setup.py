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
  description='A very basic calculator',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='https://t.me/pivo_txt',  
  author='Alexander Nikitin',
  author_email='sniknickitin@yandex.ru',
  license='MIT', 
  classifiers=classifiers,
  keywords='forcasting', 
  packages=find_packages(),
  install_requires=[''] 
)