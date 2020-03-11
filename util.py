import quandl
import math
import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats import norm

def cnd(val, alpha, sigma):
    return (1.0 + math.erf((val - alpha) / (sigma*math.sqrt(2.0))))/2.0

def csnd(point):
    return (1.0 + math.erf(point / math.sqrt(2.0))) / 2.0

def date_range_from_today(years, months, days):
    """
    Returns a matrix of data for the stock within the provided dates

    Inputs:
    -----------------
        - years:    number of years ago to get
        - months:   number of months ago to get
        - days:     number of days ago to get

    Outputs:
    -----------------
        - (startdate, enddate):     (the date years, months, and days from today;
                                        today's date)
    """

    # get date information 
    tnow = date.today()
    
    startyear = tnow.year - years
    startmonth = tnow.month - months
    startday = tnow.day - days

    enddate = tnow.isoformat()
    
    sd = tnow.replace(year=startyear, month=startmonth, day=startday)
    startdate = sd.isoformat()

    return (startdate, enddate)

class StockSeries:
    def __init__ (self, stocksym='', data=None, maxsigma=0, minsigma=0, vol=None, drift=None, cgr=None, tspans=(251, 90, 30), sd=date.today().replace(year=date.today().year - 2).isoformat(), ed=date.today().isoformat(), api_key=''):
        self.stocksym_  = stocksym
        self.data_      = data
        self.maxsigma_  = maxsigma
        self.minsigma_  = minsigma
        self.vol_       = vol
        self.drift_     = drift
        self.cgr_       = cgr
        self.tspans_    = tspans
        self.sd_        = sd
        self.ed_        = ed
        self.api_key_   = api_key

    #########################################
    #   begin setters for stock variables   #
    ######################################### 

    def set_stocksym(self, stocksym):
        self.stocksym_ = stocksym

    def set_data(self, data):
        self.data_ = data

    def set_maxsigma(self, maxsigma):
        self.maxsigma_ = maxsigma

    def set_minsigma(self, minsigma):
        self.minsigma_ = minsigma

    def set_vol(self, vol):
        self.vol_ = vol

    def set_drift(self, drift):
        self.drift_ = drift

    def set_cgr(self, cgr):
        self.cgr_ = cgr

    def set_tspans(self, tspans):
        self.tspans_ = tspans

    def ed(self, ed):
        self.ed_ = ed

    def set_sd(self, sd):
        self.sd_ = sd

    #########################################
    #   begin getters for stock variables   #
    ######################################### 

    def get_stocksym(self):
        return self.stocksym_

    def get_data(self):
        return self.data_

    def get_maxsigma(self):
        return self.maxsigma_

    def get_minsigma(self):
        return self.minsigma_

    def get_vol(self):
        return self.vol_

    def get_drift(self):
        return self.drift_

    def get_cgr(self):
        return self.cgr_

    def get_tspans(self):
        return self.tspans_
       
    def get_sd(self):
        return self.sd_
        
    def get_ed(self):
        return self.ed_

    #########################################
    #       begin  helper  functions        #
    ######################################### 

    def comp_d1(self, stockP, strikeP, rfir, sigma, daysUntilExp):
        return math.log(stockP / strikeP) + ((rfir / 365) + (sigma**2) / 2) * daysUntilExp

    def comp_duration_vol(self, sigma, daysUntilExp):
        return sigma * math.sqrt(daysUntilExp)

    def comp_cumd1(self, d1, durationVol, optionTypeScalar):
        return csnd(optionTypeScalar * (d1 / durationVol))

    def comp_cumd2(self, d1, durationVol, optionTypeScalar):
        return csnd(optionTypeScalar * (d1 / durationVol - durationVol))

    def comp_discount(self, rfir, daysUntilExp):
        return math.exp(-rfir * daysUntilExp / 365)

    def comp_price(self, stockP, strikeP, cumd1, cumd2, discount, optionTypeScalar):
        return optionTypeScalar * ((stockP * cumd1) - (strikeP * discount * cumd2))

    def comp_BSM_price_helper(self, stockP=0, strikeP=0, sigma=0, rfir=0.01, daysUntilExp=0, optionTypeScalar=1):
        d1 = self.comp_d1(stockP, strikeP, rfir, sigma, daysUntilExp)
        durationVol = self.comp_duration_vol(sigma, daysUntilExp)

        cumd1 = self.comp_cumd1(d1, durationVol, optionTypeScalar)
        cumd2 = self.comp_cumd2(d1, durationVol, optionTypeScalar)

        discount = self.comp_discount(rfir, daysUntilExp)
        price = self.comp_price(stockP, strikeP, cumd1, cumd2, discount, optionTypeScalar)

        return price
    
    #########################################
    #     begin lookup for stock data       #
    ######################################### 

    def lookup_stock_data (self, apikey):
        """
        Returns a matrix of data for the stock within the provided dates

        Inputs:
        -----------------
            - apikey:   the quandl api key for the user
            - sd:       the date to start collecting data
            - ed:       the date to end data collection

        Outputs:
        -----------------
            - self.data_:     pandas dataframe of stock data
                             [Date, Volume, Close] 
        """

        # setup symbol for quandl lookup
        stocksymlo = self.stocksym_.lower()
        stocksymup = self.stocksym_.upper()
        stocksym = stocksymlo
        quandlstock = str("WIKI/")
        quandlstock = quandlstock + stocksymup

        # setup quandl api key
        quandl.ApiConfig.api_key = apikey

        # get quandl data for desired dates
        self.data_ = quandl.get(quandlstock, start_date=self.sd_, end_date=self.ed_)
        self.api_key_ = apikey

    def lookup_rf_rate (self):
        quandl.ApiConfig.api_key = self.api_key_

        return quandl.get("FRED/DTB3", rows=1).iat[0, 0] #get 0th value in 0th row

    #########################################
    #    begin comps for stock variables    #
    ######################################### 

    def comp_cgr (self):
        """
        Computes the continuous growth rate for a stock over its lifetime

        Inputs:
        -----------------
            - data:         pandas dataframe of stock data
                                [Date, Volume, Close]
            
        Outputs:
        -----------------
            - self.data_:     pandas dataframe of stock data
                                [Date, Julian, Day, Close, Volume, cgr] 
            - self.cgr_:      continuous growth rate of stock data
        """

        # setup stock matrix
        data = self.data_.reset_index()
        data = data[['Date','Volume','Close']]

        jdate = ((data['Date'].dt.year -2000) * 1000) + data.Date.dt.dayofyear
        data['Julian'] = jdate
        data['Day'] = data.Date.dt.weekday_name

        # compute continuous growth rate returns for each trading day
        price = np.array(data.Close)
        pricep1 = np.roll(price,1)
        lnratio = price/pricep1

        cgr = np.log(lnratio)
        cgr[0]=999.99
        data['cgr'] = cgr

        self.cgr_ = cgr
        self.data_ = data[['Date', 'Julian', 'Day', 'Close','Volume','cgr']]

    def comp_vol_drift (self):
        """
        Computes the volatility, drift, minsigma, and maxsigma for each timespan

        Inputs:
        -----------------
            - tspan:        tuple of time spans (in days) to get data for.
                                all entries must be less than stock data length
            
        Outputs:
        -----------------
            - self.data_:     pandas dataframe of stock data for tspans
                                [['Date', 'Julian', 'Day', 'Close','Volume','cgr','XSigma_<tspan[0]>','XSigma_<tspan[1]>', ...]] 
            - self.maxsigma_:     numpy array (n_spans) of max sigma per tspan
            - self.minsigma_:     numpy array (n_spans) of min sigma per tspan
            - self.drift_:        numpy array (n_spans) of drift per tspan
            - self.vol_:          numpy array (n_spans) of volatility per tspan
        """

        # setup to calculate volatility, min / max sigmas, & drift
        length = len(self.data_)
        n_spans = len(self.tspans_)

        cur_cols = ['Date', 'Julian', 'Day', 'Close','Volume','cgr'] #probs a way to use data to do this..

        vol = np.zeros(n_spans)
        drift = np.zeros(n_spans)

        maxsigma = np.zeros(n_spans)
        minsigma = np.zeros(n_spans)
        
        # compute vol, minsigma, maxsigma, drift
        xsigma = np.zeros((length,n_spans))
        for i in range(0,n_spans):
            maxsigma[i] = 0.0
            minsigma[i] = 0.0

            drift[i] = np.mean(self.cgr_[length-self.tspans_[i]:length])
            vol[i] = np.std(self.cgr_[length-self.tspans_[i]:length])

            for j in range(length-self.tspans_[i],length):
                # calculate sigma for range
                xsigma[j,i] = (self.cgr_[j] - drift[i])/vol[i]

                # set min/max sigma accordingl
                if xsigma[j,i] > maxsigma[i]:
                    maxsigma[i] = xsigma[j,i]
                if xsigma[j,i] < minsigma[i]:
                    minsigma[i] = xsigma[j,i]

            # add sigmas to stock table
            col = 'XSigma_' + str(self.tspans_[i]) 
            cur_cols += [col]
            self.data_[col] = xsigma[:,i]


        self.data_ = self.data_[cur_cols]
        self.vol_ = vol
        self.drift_ = drift
        self.minsigma_ = minsigma
        self.maxsigma_ = maxsigma

    def comp_option_prob(self, stockP, option, tspan, use_drift=False):
        """
        computes the probability that an option is ITM

        Inputs:
        -----------------
            - tspan:        time span to use for volatility and drift data
            
        Outputs:
        -----------------
            - self.data_:     pandas dataframe of stock data for tspans
                                [['Date', 'Julian', 'Day', 'Close','Volume','cgr','XSigma_<tspan[0]>','XSigma_<tspan[1]>', ...]] 
            - self.maxsigma_:     numpy array (n_spans) of max sigma per tspan
            - self.minsigma_:     numpy array (n_spans) of min sigma per tspan
            - self.drift_:        numpy array (n_spans) of drift per tspan
            - self.vol_:          numpy array (n_spans) of volatility per tspan
        """
        
        i_tspan = self.tspans_.index(tspan)
        sigma = self.vol_[i_tspan] # change this...

        # calculate duration volatility
        durationVol = math.sqrt(option.daysUntilExp_) * sigma

        requiredCGR = math.log((option.strikePrice_ + option.optionPrice_) / stockP)


        if use_drift:
            drift = self.drift_[i_tspan] # and this..

            # normalize drift term over time period
            desiredDrift = math.exp(drift * option.daysUntilExp_) - 1

            adjustedCGR = requiredCGR - desiredDrift

            cumProb = csnd(adjustedCGR / durationVol)
            prob = self.comp_prob(cumProb, option.type_)

        else:
            cumProb = csnd(requiredCGR / durationVol)
            prob = self.comp_prob(cumProb, option.type_)


        prob = self.comp_prob(cumProb, option.type_)

        return prob

    def comp_prob(self, cumProb, optionType):
        prob = 0 
        if optionType == 'call':
            prob = 1 - cumProb

        elif optionType == 'put':
            prob = cumProb

        else:
            print "Given option has invalid type. Received", optionType, "expected call or put"
            exit(1)

        return prob

    def comp_BSM_price(self, stockP, option, tspan, rfir='auto'):
        """
        """

        i_tspan = self.tspans_.index(tspan)
        sigma = self.vol_[i_tspan] # change this...?

        if rfir == 'auto':
            rfir = self.lookup_rf_rate()

        optionTypeScalar = -1 if option.type_ == 'put' else 1 # negatively scale values for put

        return self.comp_BSM_price_helper(stockP=stockP, 
                                          strikeP=option.strikePrice_, 
                                          sigma=sigma, 
                                          rfir=rfir, 
                                          daysUntilExp=option.daysUntilExp_, 
                                          optionTypeScalar=optionTypeScalar)

    def comp_BSM_iv(self, stockP, option, rfir='auto'):
        """
        """

        if rfir == 'auto':
            rfir = self.lookup_rf_rate()

        optionTypeScalar = -1 if option.type_ == 'put' else 1 # negatively scale values for put

        #add peg price stuff later

        target = option.optionPrice_
        precision = float(1e-4)
        count = 0
        low = 0.0
        high = 1.0

        tempIV = float((high + low) / 2)
        tempP = self.comp_BSM_price_helper(stockP=stockP,
                                           strikeP=option.strikePrice_,
                                           sigma=tempIV,
                                           rfir=rfir,
                                           daysUntilExp=option.daysUntilExp_,
                                           optionTypeScalar=optionTypeScalar)

        while tempP <= (target - precision) or tempP >= (target + precision):
            if tempP >= (target + precision):
                high = tempIV
            else:
                low = tempIV
            
            tempIV = float((high + low) / 2)
            tempP = self.comp_BSM_price_helper(stockP=stockP,
                                               strikeP=option.strikePrice_,
                                               sigma=tempIV,
                                               rfir=rfir,
                                               daysUntilExp=option.daysUntilExp_,
                                               optionTypeScalar=optionTypeScalar)
            count += 1
        return tempIV

    def comp_BSM_delta(self, stockP, option, rfir='auto'):
        """
        """

        if rfir == 'auto':
            rfir = self.lookup_rf_rate()

        optionTypeScalar = -1 if option.type_ == 'put' else 1 # negatively scale values for put

        td_days = option.daysUntilExp_ - 1
        sigma = self.comp_BSM_iv(stockP, option, rfir)
        durationVol = self.comp_duration_vol(sigma, option.daysUntilExp_)

        d1 = self.comp_d1(stockP, option.strikePrice_, rfir, sigma, td_days)
        delta = self.comp_cumd1(d1, durationVol, optionTypeScalar)

        return delta

    def comp_BSM_td(self, stockP, option, rfir='auto'):
        """
        """

        #
        #   Below we calculate one day time decay using our new value for volatility
        #   Possible BUG: This may not work if there is only one day left!! 
        #
        
        if rfir == 'auto':
            rfir = self.lookup_rf_rate()

        optionTypeScalar = -1 if option.type_ == 'put' else 1 # negatively scale values for put

        td_days = option.daysUntilExp_ - 1
        sigma = self.comp_BSM_iv(stockP, option, rfir)
        newP = self.comp_BSM_price_helper(stockP=stockP,
                                          strikeP=option.strikePrice_,
                                          sigma=sigma,
                                          rfir=rfir,
                                          daysUntilExp=td_days,
                                          optionTypeScalar=optionTypeScalar)
        td = option.optionPrice_ - newP

        return td

    #########################################
    #    begin print/plot/save variables    #
    ######################################### 

    def print_vol_table(self):
        coltitles = ['Volatility','MaxSigma','MinSigma','Drift']
        rowtitles = [str(tspan) for tspan in self.tspans_]

        finData = []
        for i in range(len(rowtitles)):
            finData += [[self.vol_[i], self.maxsigma_[i], self.minsigma_[i], self.drift_[i]]]

        finData = np.array(finData)

        volgrid = pd.DataFrame(finData)
        volgrid.index = rowtitles
        volgrid.columns = coltitles
        print volgrid

    def save_data_vol(self, fname):
        coltitles = ['Volatility','MaxSigma','MinSigma','Drift']
        rowtitles = [str(tspan) for tspan in self.tspans_]

        finData = []
        for i in range(len(rowtitles)):
            finData += [[self.vol_[i], self.maxsigma_[i], self.minsigma_[i], self.drift_[i]]]

        finData = np.array(finData)

        volgrid = pd.DataFrame(finData)
        volgrid.index = rowtitles
        volgrid.columns = coltitles

        filename = fname
        filename += "/hv"
        filename += str(self.stocksym_)
        filename += str(self.ed_.day) + "_"
        filename += str(self.ed_.month) + "_"
        filename += str(self.ed_.year - 2000) + ".xlsx"

        cashwriter = pd.ExcelWriter(filename)
        stock.to_excel(cashwriter,'Data')
        volgrid.to_excel(cashwriter,'Dataframe')
        cashwriter.save()

class Option:
    def __init__ (self, strikePrice=None, optionPrice=None, daysUntilExpiration=None, optionType='call'):
        self.strikePrice_ = strikePrice
        self.optionPrice_ = optionPrice # replace with bid & ask price instead & peg_coef
        self.daysUntilExp_ = daysUntilExpiration # TODO replace with an end date and a function that calculates days until exp
        self.type_ = optionType

    def get_optionPrice (self):
        return self.optionPrice_

    def get_strikePrice (self):
        return self.strikePrice_

    def get_daysUntilExp (self):
        return self.daysUntilExp_

    def get_type (self):
        return self.type_

    def set_optionPrice (self, optionPrice):
        self.optionPrice_ = optionPrice

    def set_strikePrice (self, strikePrice):
        self.strikePrice_ = strikePrice

    def set_daysUntilExp (self, daysUntilExp):
        self.daysUntilExp_ = daysUntilExp

    def set_type (self, optionType):
        self.type_ = optionType


if __name__ == '__main__':
    print "Running tests:"
    print

    stock = StockSeries(stocksym="WMT")
    print "symbol is:", stock.get_stocksym()

    print "testing get data"

    stock.lookup_stock_data('7cy5HU13M48ssSdyb4Lx')

    stock.comp_cgr()

    stock.comp_vol_drift()

    stock.print_vol_table()

    option = Option(strikePrice=91, optionPrice=1.04, daysUntilExpiration=15, optionType='put')
    print stock.comp_option_prob(92.69, option, 251, use_drift=True)

    print stock.comp_BSM_iv(92.69, option, rfir=0.0225)
    print stock.comp_BSM_td(92.69, option, rfir=0.0225)
    print stock.comp_BSM_delta(92.69, option, rfir=0.0225)

