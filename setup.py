from setuptools import find_packages, setup

setup(
    name="pair",
    version="0.1.0",
    description="PAIR: adapting history-compression prompts from verified compaction boundaries",
    packages=find_packages(include=["pair", "pair.*"]),
    python_requires=">=3.10",
    install_requires=["openai>=1.50", "jinja2>=3.1", "tiktoken>=0.9"],
)
