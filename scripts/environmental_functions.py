# environmental_response.py
class EnvironmentalFunctions:
    
    @staticmethod
    def fPAR(W):
        return max(0.0, min(1.0, 1.0 - 0.0445 * (W - 5.0)))

    @staticmethod
    def fT(x, T0, T1, T2):
        if x < T0:
            return 0.0
        elif x < T1:
            return (x - T0) / (T1 - T0)
        elif x < T2:
            return 1.0
        elif x < 40.0:
            return (40.0 - x) / (40.0 - T2)
        else:
            return 0.0

    @staticmethod
    def fW(x, PET):
        if PET <= 3.8:
            if x <= 0.2:
                return 4.0 * x
            elif x <= 0.4:
                return 0.75 * x + 0.65
            elif x <= 0.6:
                return 0.25 * x + 0.85
            else:
                return 1.0
        elif PET <= 6.5:
            if x <= 0.2:
                return 2.0 * x
            elif x <= 0.4:
                return 1.5 * x + 0.1
            elif x <= 0.6:
                return x + 0.3
            elif x <= 0.8:
                return 0.5 * x + 0.6
            else:
                return 1.0
        else:
            return x

    @staticmethod
    def fWFAO(x, PET):
        p_hat = 0.60
        p_FAO = max(0.1, min(p_hat + 0.04 * (5.0 - PET), 0.8))
        W_star = 1.0 - p_FAO
        fWval = min(1.0, x / W_star)
        return fWval

    @staticmethod
    def fWfxp(x, Wcr):
        fWval = max(min(1.0, x / Wcr), 0.0)
        return fWval

    # @staticmethod
    def SEA(x, minSEA, maxSEA, ST1, ST2):
        if x < 200.:
            return minSEA
        elif x < (ST1 - 200.):
            return minSEA + (maxSEA - minSEA) * (x - 200) / (ST1 - 400)
        elif x < (ST1 - 100.):
            return maxSEA
        elif x < ST2:
            return maxSEA + (minSEA - maxSEA) * (x - ST1 + 100) / (ST2 - ST1 + 100)
        else:
            return minSEA
        