#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init, Author

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")


name = "synergy_simulation"
version = '0.0.1'
authors = [Author('Xingmin Aaron Zhang', 'kingmanzhang@gmail.com')]
summary = 'simulate phenotype synergy for specified disease'
url = 'https://github.com/kingmanzhang/synergy_simu'


@init
def set_properties(project):
    project.depends_on('obonetx')
    project.depends_on('mutual-information')

    project.set_property('unittest_module_glob', '*_test')

    project.set_property('coverage_break_build', False)
    project.set_property('coverage_threshold_warn', 70)

    project.get_property('distutils_commands').append('bdist_wheel')
    project.set_property('distutils_console_scripts', ["syn_simu = syn_simu_runner:main "])

    project.default_task = ['clean', 'analyze', 'publish']
