from setuptools import setup

environment = [
    'pandas==1.3.4',
    'numpy==1.21.2',
    'spacy==2.1.0',
]

setup(name='Coref-medialab',
        version='0.1',
        license='MIT',
        packages=['Coref-medialabs'],
        zip_safe=False,
        install_requires=environment,
        author = "Clémentine Abed Meraim",
        description = ("A practical wrapper for neuralcoref and e2ecoref"),
        keywords = "coreference")