from setuptools import setup

setup(
    name='BeeMeterHardware',
    version='0.0',
    packages=[''],
    url='',
    license='',
    author='TS',
    author_email='schaumloeffel.tim@web.de',
    description='A tool to read out several sensors in parallel and store the data.',
    install_requires=['numpy', 'spidev', 'adafruit-blinka', 'adafruit-circuitpython-dht', 'Adafruit-GPIO', 'matplotlib',
                      'picamera']
)
