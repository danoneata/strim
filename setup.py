from setuptools import setup, find_packages

setup(
    name="transcribe-with-images",
    version="0.0.1",
    # url="https://git.speed.pub.ro/ai4trust/aletheia",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    description="Transcribe speech with images",
    packages=find_packages(),
    install_requires=["black", "click", "streamlit", "ruff"],
)
