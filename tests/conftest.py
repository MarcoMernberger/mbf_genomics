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
from pypipegraph.testing.fixtures import (  # noqa:F401
    new_pipegraph,  # noqa:F401
    pytest_runtest_makereport,  # noqa:F401
)  # noqa:F401

# from mbf_externals.testing.fixtures import local_store, global_store  # noqa:F401
root = Path(__file__).parent.parent
sys.path.append(str(root / "src"))

from plotnine.tests.conftest import (  # noqa:F401
    _setup,
    _teardown,  # noqa:F401
    pytest_assertrepr_compare,
)

_setup()


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
def both_ppg_and_no_ppg(request):
    if request.param:
        if request.cls is None:
            target_path = (
                Path(request.fspath).parent
                / "run"
                / ("." + request.node.name + str(request.param))
            )
        else:
            target_path = (
                Path(request.fspath).parent
                / "run"
                / (request.cls.__name__ + "." + request.node.name)
            )
        if target_path.exists():  # pragma: no cover
            shutil.rmtree(target_path)
        target_path = target_path.absolute()
        old_dir = Path(os.getcwd()).absolute()
        try:
            first = [False]

            def np():
                if not first[0]:
                    Path(target_path).mkdir(parents=True, exist_ok=True)
                    os.chdir(target_path)
                    Path("logs").mkdir()
                    Path("cache").mkdir()
                    Path("results").mkdir()
                    Path("out").mkdir()
                    import logging

                    h = logging.getLogger("pypipegraph")
                    h.setLevel(logging.WARNING)
                    first[0] = True

                rc = ppg.resource_coordinators.LocalSystem(1)
                ppg.new_pipegraph(rc, quiet=True, dump_graph=False)
                ppg.util.global_pipegraph.result_dir = Path("results")
                g = ppg.util.global_pipegraph
                g.new_pipegraph = np
                return g

            def finalize():
                if hasattr(request.node, "rep_setup"):

                    if request.node.rep_setup.passed and (
                        hasattr(request.node, "rep_call")
                        and (
                            request.node.rep_call.passed
                            or request.node.rep_call.outcome == "skipped"
                        )
                    ):
                        try:
                            shutil.rmtree(target_path)
                        except OSError:  # pragma: no cover
                            pass

            request.addfinalizer(finalize)
            yield np()

        finally:
            os.chdir(old_dir)
    else:
        if request.cls is None:
            target_path = (
                Path(request.fspath).parent
                / "run"
                / ("." + request.node.name + str(request.param))
            )
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
                class Dummy():
                    pass
                d = Dummy
                d.new_pipegraph = lambda: None
                return d

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
    """Clear the annotator singleton instance cache
    which is only used if no ppg is in play"""
    import mbf_genomics.annotator

    mbf_genomics.annotator.annotator_singletons.clear()


def pytest_generate_tests(metafunc):
    if "both_ppg_and_no_ppg" in metafunc.fixturenames:
        metafunc.parametrize("both_ppg_and_no_ppg", [True, False], indirect=True)
