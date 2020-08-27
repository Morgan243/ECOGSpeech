from setuptools import setup
setup(name='ECOGSpeech',
      version='0.1',
      description='Processing ECOG+Voice recordings',
      author='Morgan Stuart & Srdjan Lesaja',
      packages=['ecog_speech'],
      requires=[
            'BCI2kReader',
            'torchaudio', 'tqdm',
            'numpy', 'pandas',
            'sklearn',
            'attrs'])