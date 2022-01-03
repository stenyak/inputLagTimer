#!/usr/bin/env python
###############################################################################
# Copyright 2021 Bruno Gonzalez Campo <stenyak@stenyak.com> (@stenyak)        #
# Distributed under MIT license (see license.txt)                             #
###############################################################################

# compiles the entire pythong thing into a ready-to-run binary
import PyInstaller.__main__
PyInstaller.__main__.run(['inputLagTimer.py', '--distpath=.', '--onefile', '--icon=inputLagTimer.ico', '--windowed'])
