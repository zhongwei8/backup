#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: farmer
# @Date: 2020-04-14


class Log:
    FAIL = '\033[91m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    OK_BLUE = '\033[94m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def d(log):
        print(f'{Log.OK_BLUE}Debug: {log}{Log.ENDC}')

    @staticmethod
    def w(log):
        print(f'{Log.WARNING}Warn: {log}{Log.ENDC}')

    @staticmethod
    def i(log):
        print(f'{Log.OK_GREEN}Info: {log}{Log.ENDC}')

    @staticmethod
    def e(log):
        print(f'{Log.FAIL}Error: {log}{Log.ENDC}')
