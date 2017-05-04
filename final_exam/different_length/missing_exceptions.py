# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com


class MissingInMinOb(Exception, ):

    def __init__(self, value):
        self.value = value

    def __rpr__(self):
        return "missing at %sth, too early missing"
