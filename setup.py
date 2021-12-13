from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess


#class PostDevelopCommand(develop):
#    """Post-installation for development mode."""
#    def run(self):
        #develop.run(self)
        #process = subprocess.Popen("python -m spacy download en".split(),
        #                   stdout=subprocess.PIPE,
        #                   stderr=subprocess.STDOUT)
        #returncode = process.wait()
        #print('ping returned {0}'.format(returncode))
        #print(process.stdout.read())
        #subprocess.check_call("python -m spacy download en".split())
        #subprocess.check_call("bash install_coref.sh".split())
        
        #process = subprocess.Popen("bash install_coref.sh".split(),
        #                   stdout=subprocess.PIPE,
        #                   stderr=subprocess.STDOUT)
        #returncode = process.wait()
        #print('ping returned {0}'.format(returncode))
        #print(process.stdout.read())



#class PostInstallCommand(install):
#    """Post-installation for installation mode."""
#    def run(self):
        #install.run(self)

        #process = subprocess.Popen("python -m spacy download en".split(),
        #                   stdout=subprocess.PIPE,
        #                   stderr=subprocess.STDOUT)
        #returncode = process.wait()
        #print('ping returned {0}'.format(returncode))
        #print(process.stdout.read())
        #check_call("python -m spacy download en".split())
        #check_call("bash install_coref.sh".split())
        
        #process = subprocess.Popen("bash install_coref.sh".split(),
        #                   stdout=subprocess.PIPE,
        #                   stderr=subprocess.STDOUT)
        #returncode = process.wait()
        #print('ping returned {0}'.format(returncode))
        #print(process.stdout.read())

        
environment = [
    'pandas==1.3.4',
    'numpy==1.21.2',
    'spacy>=2.1.0,<3.0.0',
    'neuralcoref==4.0',
    'tensorflow-gpu==1.15',
    'tensorflow-hub>=0.4.0',
    'h5py',
    'nltk',
    'pyhocon',
    'scipy',
    'sklearn',
    'colorama',
]

setup(name='Coref-medialab',
        version='0.1',
        license='MIT',
        zip_safe=False,
        python_requires='==3.7.6',
        install_requires=environment,
        author = "Clémentine Abed Meraim",
        description = ("A practical wrapper for neuralcoref and e2ecoref"),
        keywords = "coreference",
        #cmdclass={
        #    "develop": PostDevelopCommand,
        #    "install": PostInstallCommand,
        #    },
     )