#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import sys
import shutil
import os
import pytest
import pypipegraph as ppg
from pathlib import Path
from pypipegraph.tests.fixtures import new_pipegraph  # noqa:F401

# from mbf_externals.tests.fixtures import local_store, global_store  # noqa:F401
root = Path(__file__).parent.parent
sys.path.append(str(root / "src"))


@pytest.fixture
def no_pipegraph(request):
    """No pipegraph, but seperate directory per test"""
    if request.cls is None:
        target_path = Path(request.fspath).parent / "run" / ("." + request.node.name)
    else:
        target_path = (
            Path(request.fspath).parent
            / "run"
            / (request.cls.__name__ + "." + request.node.name)
        )
    if target_path.exists():  # pragma: no cover
        shutil.rmtree(target_path)
    target_path = target_path.absolute()
    target_path.mkdir()
    old_dir = Path(os.getcwd()).absolute()
    os.chdir(target_path)
    try:

        def np():
            ppg.util.global_pipegraph = None
            return None

        def finalize():
            if hasattr(request.node, "rep_setup"):

                if request.node.rep_setup.passed and (
                    request.node.rep_call.passed
                    or request.node.rep_call.outcome == "skipped"
                ):
                    try:
                        shutil.rmtree(target_path)
                    except OSError:  # pragma: no cover
                        pass

        request.addfinalizer(finalize)
        ppg.util.global_pipegraph = None
        yield np()

    finally:
        os.chdir(old_dir)


@pytest.fixture
def clear_annotators(request):
    """No pipegraph, but seperate directory per test"""
    import mbf_genomics.annotator

    mbf_genomics.annotator.annotator_singletons.clear()
