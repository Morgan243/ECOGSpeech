from setuptools import setup
setup(name='ECOGSpeech',
      version='0.1',
      description='Processing ECOG+Voice recordings',
      author='Morgan Stuart & Srdjan Lesaja',
      packages=['ecog_speech'],
      requires=[
            'torch',  # Probably make sure installed using torch website instructions for the platform
            'torchaudio',
            'tqdm',
            'numpy', 'pandas', 'scipy',
            'scikit-learn', 'mne',
            'attrs',
            'python_speech_features',
      ])
