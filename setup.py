import setuptools

# Use ./README.md as the `long_description`
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup packaging
setuptools.setup(
    name="openisr-iLoveDataJjia",
    version="0.0.1",
    author="iLoveDataJjia",
    author_email="ilovedata.jjia@gmail.com",
    description="🚀🔥 OpenISR is an image super resolution lightweight SaaS web application. It is powered by OpenCV & PyTorch and has the purpose to upscale your images without losing quality! It is also optimized for scalability and easily deployable thanks to Celery & Docker!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iLoveDataJjia/openisr",
    project_urls={
        "Bug Tracker": "https://github.com/iLoveDataJjia/openisr/issues"},
    classifiers=[
        "Programming Language :: Python :: 3.7.11",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS-independent"],
    package_dir={"": "openisr"},
    packages=setuptools.find_packages(where="openisr"),
    python_requires="==3.7.11",
)