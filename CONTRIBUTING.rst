Contributions
=============

Contributions are welcome, greatly appreciated and overall helps to improve the package.

Report Bugs
-----------

Before creating a new issue, do check if a `bug was already reported <https://github.com/zillow/luminaire/labels/bug>`__

Report bugs through `GitHub Issues <https://github.com/zillow/luminaire/issues>`__:

- Create a ``New Issue``
- Provide detailed information about the bug and preferably part of the code that exhibits the problem
- Make sure to tag label as ``bug``


Fix Bugs
--------

Look through our `GitHub issues labeled "kind:bug"
<https://github.com/zillow/luminaire/labels/bug>`__ for bugs.

Issues which are unassigned can be owned to implement it:

- Create branch name with prefix ``bugfix/<bug-type>``
- Add relevant code fix and test case
- Update the package version inside ``setup.py``
- Once ready for review create a PR for ``master`` branch


Report Features
---------------

Before creating a new feature request, do check if a `features was already requested <https://github.com/zillow/luminaire/labels/feature>`__

Report new features through `GitHub Issues <https://github.com/zillow/luminaire/issues>`__:

- Create a ``New Issue``
- Provide detailed information about the feature, how it would work, benefits and scope
- Make sure to tag label as ``feature``


Implement Features
------------------

Look through the `GitHub issues labeled "kind:feature"
<https://github.com/zillow/luminaire/labels/feature>`__ for features.

Issues which are unassigned can be owned to implement it:

- Create branch name with prefix ``feature/<feature-type>``
- Add relevant code and test case
- Update the package version inside ``setup.py``
- Once ready for review create a PR for ``master`` branch.


Github Workflow (CI/CD)
------------------------

Github Workflow is used for CI/CD.

Luminaire CI:

One each commit `Luminaire CI Workflow <https://github.com/zillow/luminaire/actions?query=workflow%3A%22Luminaire+CI%22>`__ is invoked.
Details about the Luminaire CI Workflow can be found `here <https://github.com/zillow/luminaire/blob/master/.github/workflows/python-app.yml>`__


Luminaire CD:

For releasing the code and making it available on PyPI, follow this steps:

- ``setup.py`` has a new version specified
- PR for ``master`` branch is approved and merged
- `Create a new release <https://github.com/zillow/luminaire/releases/new>`__
- Specify the ``Tag version`` with prefix ``v<version>`` example: v0.1.0
- Select ``master`` branch
- Provide relevant release title and description
- If this is a pre-release check the box
- Click ``Publish release`` and this will trigger `Luminaire CD Workflow <https://github.com/zillow/luminaire/blob/master/.github/workflows/python-publish.yml>`__
- Check `PyPI release history <https://pypi.org/project/luminaire/#history>`__


Submit Feedback
---------------

The best way to send feedback is by `creating an issue on GitHub <https://github.com/zillow/luminaire/issues>`__.

- Create a ``New Issue``
- Provide detailed information about feedback
- Make sure to tag label as ``feedback``
