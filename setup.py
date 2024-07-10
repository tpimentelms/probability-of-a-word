from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='wordsprobability',
      version='0.17',
      description='Method to get a words probability with fixes from How to Compute the Probability of a Word.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='language modelling surprisal word tokenisation probability',
      url='https://github.com/tpimentelms/probability-of-a-word',
      author='Tiago Pimentel and Clara Meister',
      author_email='tpimentelms@gmail.com',
      license='MIT',
      packages=['wordsprobability', 'wordsprobability.models', 'wordsprobability.utils'],
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'wordsprobability=wordsprobability.main:main',
          ],
      },
      install_requires=[
          'torch',
          'transformers',
          'pandas',
      ],
      zip_safe=False)