"""Teaching adaptation of main_simulation_historical_vSnow_AutoCut.py.

Original ModVege daily growth, senescence, water balance and snow equations kept.
Changes: explicit manual cuts, no fixed paths, DataFrame result, no file output.
Units: biomass and harvested biomass kg DM/ha; LAI m2/m2; Ta degrees C.
"""
from pathlib import Path
import calendar
import numpy as np
import pandas as pd
from .environmental_functions import EnvironmentalFunctions

class Simulation:
    
    def __init__(
        self, weather, model_parameters, initial_conditions, *,
        year, cut_days, cut_height_cm=5.0,
        functional_group_weights=None, ni=1.0, whc=80.0,
        temp_offset=0.0, fW_ind="fW", w_cr=0.4, const_wr=False,
    ):
        self.year = int(year)
        self.cut_days = set(int(d) for d in cut_days)
        self.cut_height_cm = float(cut_height_cm)
        self.FunctionalGroup = "FG_weighted"
        self.FunctionalGroup_weights = functional_group_weights or {
            "FG_A": 0.5, "FG_B": 0.5, "FG_C": 0.0, "FG_D": 0.0,
        }
        self.NI = float(ni)
        self.WHC = float(whc)
        self.Ta_correction = float(temp_offset)
        self.fW_ind = fW_ind
        self.W_cr = float(w_cr)
        self.const_WR = bool(const_wr)
        self.weather = weather
        self.model_parameters = model_parameters
        self.initial_conditions = initial_conditions

    def run_simulation(self):
        MeteoData = self.weather.copy()
        if "DOY" in MeteoData.columns:
            MeteoData = MeteoData.sort_values("DOY").set_index("DOY")
        # Historische Daten: meist 365 Modell-Tage, auch in Schaltjahren.
        # Laufendes Jahr: nur die tatsächlich vorhandenen Tage simulieren.
        n_days = len(MeteoData)
        max_days = 366 if calendar.isleap(self.year) else 365
        if not 1 <= n_days <= max_days or MeteoData.index.tolist() != list(range(1, n_days + 1)):
            raise ValueError(
                f"Incomplete DOY series for {self.year}: "
                f"expected consecutive DOY 1..{n_days} (maximum {max_days})"
            )
        for col in ("ETc", "Precip", "PAR", "Ta"):
            if col not in MeteoData or not np.isfinite(MeteoData[col]).all():
                raise ValueError(f"Missing or invalid weather data in column {col}")
        MeteoData.index = np.arange(len(MeteoData))

        ModelParameters = self.model_parameters.copy()
        weights = pd.Series(self.FunctionalGroup_weights, dtype=float)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("Functional-group weights must sum to 1")
        ModelParameters["FG_weighted"] = ModelParameters[weights.index].astype(float).mul(weights).sum(axis=1)
        InitialConditions = self.initial_conditions.copy()

        ndays = len(MeteoData.index)

        # initial conditions
        BMGV0 = InitialConditions.loc["BMGV"].values[0]
        BMGR0 = InitialConditions.loc["BMGR"].values[0]
        BMDR0 = InitialConditions.loc["BMDR"].values[0]
        BMDV0 = InitialConditions.loc["BMDV"].values[0]
        AgeGV0 = InitialConditions.loc["AgeGV"].values[0]
        AgeGR0 = InitialConditions.loc["AgeGR"].values[0]
        AgeDV0 = InitialConditions.loc["AgeDV"].values[0]
        AgeDR0 = InitialConditions.loc["AgeDR"].values[0]
        SENGV0 = InitialConditions.loc["SENGV"].values[0]
        SENGR0 = InitialConditions.loc["SENGR"].values[0]
        ABSDV0 = InitialConditions.loc["ABSDV"].values[0]
        ABSDR0 = InitialConditions.loc["ABSDR"].values[0]
        ST0 = InitialConditions.loc["ST"].values[0]
        cBM0 = InitialConditions.loc["cBM"].values[0]

        # Model Parameters (FG)
        SLA = ModelParameters[self.FunctionalGroup].loc["SLA"]
        pcLAM = ModelParameters[self.FunctionalGroup].loc["pcLAM"]
        T0 = ModelParameters[self.FunctionalGroup].loc["T0"]
        T1 = ModelParameters[self.FunctionalGroup].loc["T1"]
        T2 = ModelParameters[self.FunctionalGroup].loc["T2"]
        RUEmax = ModelParameters[self.FunctionalGroup].loc["RUEmax"]
        ST1 = ModelParameters[self.FunctionalGroup].loc["ST1"]
        ST2 = ModelParameters[self.FunctionalGroup].loc["ST2"]
        minSEA = ModelParameters[self.FunctionalGroup].loc["minSEA"]
        maxSEA = ModelParameters[self.FunctionalGroup].loc["maxSEA"]
        LLS = ModelParameters[self.FunctionalGroup].loc["LLS"]
        KGV = ModelParameters[self.FunctionalGroup].loc["KGV"]
        KGR = ModelParameters[self.FunctionalGroup].loc["KGR"]
        KlDV = ModelParameters[self.FunctionalGroup].loc["KlDV"]
        KlDR = ModelParameters[self.FunctionalGroup].loc["KlDR"]
        BDGV = ModelParameters[self.FunctionalGroup].loc["BDGV"]
        BDDV = ModelParameters[self.FunctionalGroup].loc["BDDV"]
        BDGR = ModelParameters[self.FunctionalGroup].loc["BDGR"]
        BDDR = ModelParameters[self.FunctionalGroup].loc["BDDR"]
        sigmaGV = ModelParameters[self.FunctionalGroup].loc["sigmaGV"]
        sigmaGR = ModelParameters[self.FunctionalGroup].loc["sigmaGR"]

        # Meteo Data
        PET = MeteoData["ETc"]
        PP = MeteoData["Precip"]
        PAR = MeteoData["PAR"]
        Ta = MeteoData["Ta"]

        # -- Declare state variables
        BMGV = [np.nan] * ndays
        BMGR = [np.nan] * ndays
        BMDV = [np.nan] * ndays
        BMDR = [np.nan] * ndays
        BM   = [np.nan] * ndays
        cBM  = [np.nan] * ndays
        dBM  = [np.nan] * ndays
        hvBM = [np.nan] * ndays
        ST   = [np.nan] * ndays
        PGRO = [np.nan] * ndays
        GRO  = [np.nan] * ndays
        LAI  = [np.nan] * ndays
        LAIGV= [np.nan] * ndays
        AET  = [np.nan] * ndays
        WR   = [np.nan] * ndays
        ENV  = [np.nan] * ndays
        ENVfPAR = [np.nan] * ndays
        ENVfT   = [np.nan] * ndays
        ENVfW   = [np.nan] * ndays
        SEA_ls  = [np.nan] * ndays

        STcut = 1
        dtI = 1.0

        # Temperatur-Korrektur
        Ta = Ta + self.Ta_correction

        # Schnee-Modell (SWE)
        Ta_arr = Ta.values if isinstance(Ta, pd.Series) else np.array(Ta)
        PP_arr = PP.values if isinstance(PP, pd.Series) else np.array(PP)
        snow_melt_rate = 3.0  # mm/°C/Tag
        snow_store = 0.0
        snow_flag = np.zeros(ndays, dtype=np.uint8)
        PP_eff = np.zeros(ndays)

        for j in range(ndays):
            if Ta_arr[j] < 0.0:
                snow_store += PP_arr[j]
                PP_eff[j] = 0.0
            else:
                melt = max(0.0, snow_melt_rate * Ta_arr[j])
                actual_melt = min(snow_store, melt)
                snow_store -= actual_melt
                PP_eff[j] = PP_arr[j] + actual_melt
            snow_flag[j] = 1 if snow_store > 0.0 else 0

        # Grünlandtemperatursumme (GTS), wie im ursprünglichen ModVege-Skript:
        # Jan 0.5, Feb 0.75, März bis Juni 1.0; nur positive Ta, ohne Schnee.
        # Wachstumsstart: erste Überschreitung von 200 °C·d.
        jSGS = None
        cum_Ta = 0.0
        GTS = np.zeros(ndays, dtype=float)
        for j in range(ndays):
            if snow_flag[j] == 0 and Ta[j] > 0:
                if j < 31:
                    cum_Ta += Ta[j] * 0.5
                elif j < 59:
                    cum_Ta += Ta[j] * 0.75
                elif j < 180:
                    cum_Ta += Ta[j]
            GTS[j] = cum_Ta
            if jSGS is None and cum_Ta > 200.0:
                jSGS = j + 1


        CUT_flag = np.zeros(ndays, dtype=np.uint8)
        stubble_h = self.cut_height_cm / 100.0  # cm -> m

        DOY = np.arange(0, ndays)
        for d in DOY:

            if d == 0:
                AgeGV = AgeGV0; AgeGR = AgeGR0; AgeDV = AgeDV0; AgeDR = AgeDR0
                SENGV = SENGV0; SENGR = SENGR0
                ABSDV = ABSDV0; ABSDR = ABSDR0

            BMGVp = BMGV0 if d == 0 else BMGV[d - 1]
            BMGRp = BMGR0 if d == 0 else BMGR[d - 1]
            BMDRp = BMDR0 if d == 0 else BMDR[d - 1]
            BMDVp = BMDV0 if d == 0 else BMDV[d - 1]
            STp   = ST0   if d == 0 else ST[d - 1]
            cBMp  = cBM0  if d == 0 else cBM[d - 1]
            WRp   = self.WHC if d == 0 else WR[d - 1]

            # --- Schnee-Short-Circuit ------------------------------------------------
            if snow_flag[d] == 1:
                AET[d] = 0.0
                if not self.const_WR:
                    Drainage = max(0.0, WRp + PP_eff[d] - AET[d] - self.WHC)
                    WR[d] = max(0.0, min(self.WHC, WRp + PP_eff[d] - Drainage - AET[d]))
                else:
                    WR[d] = self.WHC

                ENVfPAR[d] = 0.0
                ENVfT[d]   = 0.1
                ENVfW[d]   = EnvironmentalFunctions.fW(WR[d]/self.WHC, PET[d]) if self.fW_ind=="fW" else \
                             (EnvironmentalFunctions.fWFAO(WR[d]/self.WHC, PET[d]) if self.fW_ind=="fWFAO" \
                              else EnvironmentalFunctions.fWfxp(WRp/self.WHC, self.W_cr))
                ENV[d] = self.NI * ENVfPAR[d] * ENVfT[d] * ENVfW[d]

                ST[d]   = STp
                PGRO[d] = 0.0
                GRO[d]  = 0.0
                SEA_ls[d] = EnvironmentalFunctions.SEA(ST[d], minSEA, maxSEA, ST1, ST2)

                BMGV[d] = BMGVp; BMGR[d] = BMGRp; BMDV[d] = BMDVp; BMDR[d] = BMDRp
                BM[d]  = BMGV[d] + BMGR[d] + BMDV[d] + BMDR[d]
                cBM[d] = cBMp
                dBM[d] = 0.0
                hvBM[d] = 0.0 if d == 0 else hvBM[d-1]
                LAI[d]   = SLA*pcLAM*(BMGV[d] + BMGR[d])/10.0
                LAIGV[d] = SLA*pcLAM*(BMGV[d])/10.0
                continue
            # -------------------------------------------------------------------------

            # Wasser/Flüsse
            LAI_et = SLA*pcLAM*(BMGVp + BMGRp)/10.0
            PTr = PET[d]*(1 - np.exp(-0.6*LAI_et))
            PEv = PET[d] - PTr

            if self.fW_ind == "fW":
                ATr = PTr * EnvironmentalFunctions.fW(WRp/self.WHC, PTr)
            elif self.fW_ind == "fWFAO":
                ATr = PTr * EnvironmentalFunctions.fWFAO(WRp/self.WHC, PTr)
            else:  # fWfxp
                ATr = PTr * EnvironmentalFunctions.fWfxp(WRp/self.WHC, self.W_cr)

            soil_rel = max(0.0, min(1.0, WRp / self.WHC))
            AEv = PEv * soil_rel
            AET[d] = ATr + AEv

            if not self.const_WR:
                Drainage = max(0.0, WRp + PP_eff[d] - AET[d] - self.WHC)
                WR[d] = max(0.0, min(self.WHC, WRp + PP_eff[d] - Drainage - AET[d]))
            else:
                WR[d] = self.WHC

            # Umweltfaktoren
            ENVfPAR[d] = EnvironmentalFunctions.fPAR(PAR[d])
            ENVfT[d]   = EnvironmentalFunctions.fT(Ta[d], T0, T1, T2)
            if self.fW_ind == "fW":
                ENVfW[d] = EnvironmentalFunctions.fW(WR[d]/self.WHC, PET[d])
            elif self.fW_ind == "fWFAO":
                ENVfW[d] = EnvironmentalFunctions.fWFAO(WR[d]/self.WHC, PET[d])
            else:
                ENVfW[d] = EnvironmentalFunctions.fWfxp(WRp/self.WHC, self.W_cr)
            ENV[d] = self.NI * ENVfPAR[d] * ENVfT[d] * ENVfW[d]

            # ST/PGRO/GRO
            LAI_gr = SLA*pcLAM*(BMGVp/10.0)
            ST[d] = STp if (jSGS is not None and DOY[d] < jSGS) else STp + max(Ta[d], 0.0) * dtI

            if jSGS is None:
                PGRO[d] = 0.0
            else:
                PGRO[d] = 0.0 if DOY[d] < jSGS else PAR[d] * RUEmax * (1. - np.exp(-0.6 * LAI_gr)) * 10.0

            SEA_now = EnvironmentalFunctions.SEA(ST[d], minSEA, maxSEA, ST1, ST2)
            SEA_ls[d] = SEA_now
            GRO[d] = PGRO[d] * ENV[d] * SEA_now

            # Repro-Anteil
            CUT_pheno = 1 if (STcut == 1 and ST[d] >= ST1 and ST[d] <= ST2) else 0
            # REP ist ein Biomasseanteil und darf bei NI > 1 nicht über 1 steigen.
            REP = np.clip((0.25 + (0.75*(self.NI - 0.35)) / 0.65) * CUT_pheno, 0.0, 1.0)
            GROGV = GRO[d]*(1 - REP)
            GROGR = GRO[d]*REP

            # Alterung/Seneszenz/Abszission
            if BMGVp - SENGV + GROGV != 0:
                dAgeGV = ((BMGVp - SENGV) / (BMGVp - SENGV + GROGV)) * (AgeGV + max(0, Ta[d])) - AgeGV
            else:
                dAgeGV = -AgeGV
            AgeGV += dAgeGV*dtI

            if BMGRp - SENGR + GROGR != 0:
                dAgeGR = ((BMGRp - SENGR) / (BMGRp - SENGR + GROGR)) * (AgeGR + max(0, Ta[d])) - AgeGR
            else:
                dAgeGR = -AgeGR
            AgeGR += dAgeGR*dtI

            fAgeGV = 1 if AgeGV/LLS < 1./3. else (3.*AgeGV/LLS if AgeGV/LLS < 1. else 3)
            fAgeGR = 1 if AgeGR/(ST2-ST1) < 1./3. else (3.*AgeGR/(ST2-ST1) if AgeGR/(ST2-ST1) < 1. else 3)

            if Ta[d] > T0:
                SENGV = KGV * BMGVp * Ta[d] * fAgeGV
            elif Ta[d] > 0:
                SENGV = 0
            else:
                SENGV = -KGV * BMGVp * Ta[d]

            if Ta[d] > T0:
                SENGR = KGR * BMGRp * Ta[d] * fAgeGR
            elif Ta[d] > 0:
                SENGR = 0
            else:
                SENGR = -KGR * BMGRp * Ta[d]

            if BMDVp - ABSDV + SENGV != 0:
                dAgeDV = ((BMDVp - ABSDV) / (BMDVp - ABSDV + SENGV)) * (AgeDV + max(0, Ta[d])) - AgeDV
            else:
                dAgeDV = -AgeDV
            AgeDV += dAgeDV*dtI

            if BMDRp - ABSDR + SENGR != 0:
                dAgeDR = ((BMDRp - ABSDR) / (BMDRp - ABSDR + SENGR)) * (AgeDR + max(0, Ta[d])) - AgeDR
            else:
                dAgeDR = -AgeDR
            AgeDR += dAgeDR*dtI

            fAgeDV = 1 if AgeDV/LLS < 1./3. else (2 if AgeDV/LLS < 2./3. else 3)
            fAgeDR = 1 if AgeDR/(ST2-ST1) < 1./3. else (2 if AgeDR/(ST2-ST1) < 2./3. else 3)

            ABSDV = KlDV*BMDVp*Ta[d]*fAgeDV if Ta[d] > 0. else 0.
            ABSDR = KlDR*BMDRp*Ta[d]*fAgeDR if Ta[d] > 0. else 0.

            # Manuelle Schnitte: DOY des ausgewählten Wetterjahres.
            did_cut_today = (d + 1) in self.cut_days
            if did_cut_today:
                if STcut == 1 and ST[d] >= ST1 and ST[d] <= ST2:
                    STcut = 0

                BMGV[d] = min(BMGVp, stubble_h * 10.0 * BDGV)
                BMDV[d] = min(BMDVp, stubble_h * 10.0 * BDDV)
                BMGR[d] = min(BMGRp, stubble_h * 10.0 * BDGR)
                BMDR[d] = min(BMDRp, stubble_h * 10.0 * BDDR)

                harvest = ((BMGVp - BMGV[d]) + (BMGRp - BMGR[d])
                           + (BMDVp - BMDV[d]) + (BMDRp - BMDR[d]))
                hvBM[d] = (0.0 if d == 0 else hvBM[d-1]) + harvest
                CUT_flag[d] = 1
                LAI[d] = SLA * pcLAM * (BMGV[d] + BMGR[d]) / 10.0
                LAIGV[d] = SLA * pcLAM * BMGV[d] / 10.0
                BM[d] = BMGV[d] + BMGR[d] + BMDV[d] + BMDR[d]
                cBM[d] = cBMp
                dBM[d] = 0.0

            # --- Normales Wachstum, wenn nicht geschnitten -------------------------
            if not did_cut_today:
                dBMGV = GROGV - SENGV
                BMGV[d] = max(BMGVp + dBMGV, BDGV*0.02*10.0)

                dBMGR = GROGR - SENGR
                BMGR[d] = max(BMGRp + dBMGR, BDGR*0.02*10.0)

                dBMDV = (1.0 - sigmaGV) * SENGV - ABSDV
                BMDV[d] = BMDVp + dBMDV

                dBMDR = (1.0 - sigmaGR) * SENGR - ABSDR
                BMDR[d] = BMDRp + dBMDR

                BM[d]  = BMGV[d] + BMGR[d] + BMDV[d] + BMDR[d]
                cBM[d] = cBMp + dBMGV + dBMGR + dBMDV + dBMDR
                dBM[d] = max(0.0, dBMGV + dBMGR + dBMDV + dBMDR)
                hvBM[d] = 0.0 if d == 0 else hvBM[d-1]

                LAI[d]   = SLA*pcLAM*(BMGV[d] + BMGR[d])/10.0
                LAIGV[d] = SLA*pcLAM*(BMGV[d])/10.0
            # -----------------------------------------------------------------------

        result = pd.DataFrame({
            "year": self.year,
            "DOY": DOY + 1,
            "BMGV": BMGV, "BMGR": BMGR, "BMDV": BMDV, "BMDR": BMDR,
            "BM": BM, "cBM": cBM, "dBM": dBM, "hvBM": hvBM,
            "ST": ST, "GTS": GTS, "PGRO": PGRO, "GRO": GRO,
            "LAI": LAI, "LAIGV": LAIGV,
            "PET": PET.to_numpy(), "AET": AET, "WR": WR,
            "ENV": ENV, "fPAR": ENVfPAR, "fT": ENVfT,
            "fW": ENVfW, "SEA": SEA_ls,
            "SGS": jSGS if jSGS is not None else np.nan,
            "snow": snow_flag, "CUT": CUT_flag,
            "Ta": Ta.to_numpy(), "PAR": PAR.to_numpy(), "Precip": PP.to_numpy(),
            "water_input": PP_eff,  # Niederschlag + Schneeschmelze, tatsächlich dem Boden zugeführt
        })
        result["cut_yield"] = result["hvBM"].diff().fillna(result["hvBM"]).clip(lower=0)
        result["f_abs"] = 1.0 - np.exp(-0.6 * result["LAI"])
        result["date"] = pd.to_datetime(self.year * 1000 + result["DOY"], format="%Y%j")
        return result


def load_weather(path: Path) -> pd.DataFrame:
    """Tab-separated original MeteoStation files with DOY, Ta, PAR, Precip, ETc."""
    return pd.read_csv(path, sep="\t")


def load_parameters(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pd.read_csv(
        root / "data/input/model_parameters/ModelParameters_R_Distr_01.csv",
        sep=";", index_col="Parameter",
    )
    initial = pd.read_csv(
        root / "data/input/initial_conditions/InitialConditions_R_Distr_01.csv",
        sep=";", index_col="Variable",
    )
    return model, initial


def simulate(weather: pd.DataFrame, year: int, cut_days: list[int],
             model_parameters: pd.DataFrame, initial_conditions: pd.DataFrame,
             *, cut_height_cm: float = 5.0, ni: float = 1.0,
             whc: float = 80.0, temp_offset: float = 0.0,
             functional_group_weights: dict | None = None) -> pd.DataFrame:
    """Simulate available consecutive weather days; biomass and harvest in kg DM/ha."""
    return Simulation(
        weather, model_parameters, initial_conditions,
        year=year, cut_days=cut_days, cut_height_cm=cut_height_cm,
        ni=ni, whc=whc, temp_offset=temp_offset,
        functional_group_weights=functional_group_weights,
    ).run_simulation()
