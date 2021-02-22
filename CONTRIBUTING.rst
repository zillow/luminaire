Contributions
=============

Contributions are welcome, greatly appreciated and overall helps to improve the package.

Submitting Bugs, Features, and other Feedback
---------------------------------------------

We use `Github Issues <https://github.com/zillow/luminaire/issues>`_ to track all kinds of feedback around Luminaire. Before creating a new issue, please seaarch through the list to see if your bug or feature has already been reported. If not, feel free to open a new one.

For bugs, please include:
- As much detailed information as you have about the bug itself
- A minimal code example that reproduces the bug behavior
- Your python version (``python --version``)
- Versions for all packages in use (``pip list``) - we need this because Luminaire makes use of on multiple packages for statistical modeling.
- Make sure to label the issues as ``bug``

For new features and enhancements, please include:
- An explanation of the feature
- How it would work (code examples would be great)
- Benefits and scope
- If your request involves a statistical method that's not currently in use by Luminaire, be sure to link to or cite scientific publications so that we can evaluate the method.

Fixing bugs and implementing features
-------------------------------------

Look through our `issue board <https://github.com/zillow/luminaire/issues>`_ with either ``kind:bug`` or ``kind:feature``. We prefer every pull request to include the following:

- A link to the issue being fixed (please use Github's syntax like "Closes #number")
- Code changes itself
- Test cases
- Documentation changes if there are any behavior or function, class, or object signature changes

Please do not bump the version number in ``setup.py`` unless asked by a project owner. 

Updating Documentation
----------------------

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`__ to compile documentation for this package, 
which is hosted on `Github pages <https://zillow.github.io/luminaire>`__

All the source code can be found under ``luminaire/docs``. Whenever a code change is made, documentation
should be updated accordingly.

Whenever a PR is merged to ``master``, Luminaire Docs workflow kicks in and
pushes the build files to ``gh-pages`` branch.


Github Workflow (CI/CD)
------------------------

Github Workflow is used for CI/CD.

Luminaire CI:

Each commit invokes the `Luminaire CI Workflow <https://github.com/zillow/luminaire/actions?query=workflow%3A%22Luminaire+CI%22>`__, which
attempts to build the package and run tests. Definition for the workflow can be found `here <https://github.com/zillow/luminaire/blob/master/.github/workflows/python-app.yml>`__

Luminaire CD:

For releasing code and making it available on PyPI. This can only be done by project owners.

- Make sure ``setup.py`` has a new version specified. If all changes were submitted by the open source community, you may need to make a new PR to bump version numbers.
- `Create a new release <https://github.com/zillow/luminaire/releases/new>`__
- Set the ``Tag version`` to the new version number with prefix ``v<version>`` example: ``v0.1.0``, and title to the same version number but without the prefix, such as ``0.1.0``
- Check that you're releasing the correct branch (usually ``master``)
- The description should include all changes included in that version. Use the commit history to help you compile it.
- If this is a pre-release check the box (pre-release version numbers should look like ``0.1.0.dev1``

Publishing the release triggers `Luminaire CD Workflow <https://github.com/zillow/luminaire/blob/master/.github/workflows/python-publish.yml>`__. This will build the package and make it available on PyPI. You can verify that the release went through by checking the `release history <https://pypi.org/project/luminaire/#history>`__
