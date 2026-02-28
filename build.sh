# python setup.py clean --all
# python3 setup.py bdist_wheel > build.log 2>&1
pip install -v --no-build-isolation -e . > build.log 2>&1
