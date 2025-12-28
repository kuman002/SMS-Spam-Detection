from setuptools import find_packages, setup

setup(
    name='sms_spam_detection',
    version='0.0.1',
    author='Kumar',
    author_email='kumar@example.com',
    packages=find_packages(),
    install_requires=[
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib'
    ]
)
