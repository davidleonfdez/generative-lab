This folder is added to the python path in conftest.py, making
any module inside available to all tests.
To avoid naming conflicts with production code, the convention
is to make the name of every helper module start with "testing_".
