from setuptools import setup, find_packages

setup(name='waqaqc',
      version='0.1',
      author="Guilherme Couto",
      author_email="gcouto@aip.de",
      description="",
      url="",
      # packages=['waqaqc'],
      packages=find_packages(),
      include_package_data=True,
      package_data={
          "waqaqc.data": ["*.dat"],
      },
      )
