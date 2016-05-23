# -*- coding: utf-8 -*-
# author: rw
# E-mail: weiyanjie10@gmail.com
from missing_exceptions import MissingInMinOb

class PredicterMixin(object, ):

    def predict_and_fit(self, ob_x):
        if hasattr(self, "min_ob"):
            min_ob = self.min_ob
        else:
            min_ob = 0
        if hasattr(self, "no_missing_in_min_ob"):
            no_missing = self.no_missing_in_min_ob
        else:
            no_missing = False
        if hasattr(self, "impute"):
            impute = self.impute
        else:
            impute = False
            
        if len(self.xs) < min_ob:
            if no_missing and ob_x == '*':
                raise MissingInMinOb(len(self.xs))
            self.xs.append(ob_x)
            self.errors.append(ob_x)
        else:
            pre_x = self.predict()
            if ob_x == '*':
                self.errors.append(0)
                if impute:
                    self.xs.append(pre_x)
                else:
                    self.xs.append(ob_x)
            else:
                self.errors.append(pre_x - ob_x)
                self.fit(pre_x, ob_x)
                self.xs.append(ob_x)
