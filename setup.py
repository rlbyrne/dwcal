from setuptools import setup

setup(
    name="DWCal",
    author="Ruby Byrne",
    author_email="rbyrne@caltech.edu",
    url="https://github.com/rlbyrne/dwcal",
    scripts=["dwcal/delay_weighted_cal.py", "dwcal/dwcal_tests.py"],
)
